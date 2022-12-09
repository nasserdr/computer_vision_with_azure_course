import torch
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import gc
from scipy.sparse import coo_matrix
from collections import OrderedDict as Dict
from tqdm import tqdm
import sys
import os 
import cv2
from .eval_utils import my_confusion_matrix, rates2metrics, cm2rates
# setting path
sys.path.append(os.getcwd() + '../dataset')

from .StreamInference import InferenceDataset
import torchvision.transforms as T
import numpy as np

class Inference():
    
    def __init__(self, model, dataset_info, save_out = False):
        # Define predictions averaging kernel
        self.kernel = dataset_info.get_inference_kernel()
        self.dataset_info = dataset_info
        self.device = dataset_info.device
        self.model = model
        
        self.patch_size = dataset_info.patch_size
        self.padding = dataset_info.padding
        self.stride = self.patch_size - self.padding
        self.save_output = save_out
    
    def getPredict(a):
        pred =a['out'].data.cpu().numpy()
        return np.argmax(pred, axis=1)[0]

    def getSP(mask):
        return np.sum(mask==1)/(mask.shape[0]*mask.shape[1])

    def getOSR(mask):
        return np.sum(mask==2)/(mask.shape[0]*mask.shape[1])
    
    def _infer_sample(self, inputs, coordinates, target, criterion = False): # H, W
        self.height, self.width = target.shape
        # initialize accumulators
        output = np.zeros((self.dataset_info.n_classes, self.height, self.width), dtype=np.float32)
        counts = np.zeros((self.height, self.width), dtype=np.float32)
        
        ds = list(zip(inputs, [torch.from_numpy(c) for c in coordinates]))
        dataloader = torch.utils.data.DataLoader(ds, batch_size=4)
        val_losses = []

        # iterate over batches of small patches
        for data, coords in dataloader:
            coords = coords.numpy()
            n_targets = torch.stack([target[c[0]:c[0]+self.patch_size,c[1]:c[1]+self.patch_size] for c in coords])
            n_targets = n_targets.to(self.device)
            # get the prediction for the batch
            data = data.to(self.device)
            with torch.no_grad():
                t_output = self.model(data)
                if criterion:
                    val_loss = criterion(t_output['out'], n_targets.long())
                    val_losses.append(val_loss.item())
                a_output = torch.softmax(t_output['out'].cpu(), dim = 1).numpy()
            # accumulate the batch predictions
            for j in range(a_output.shape[0]):
                x, y =  coords[j]
                output[:, x:(x+self.patch_size), y:(y+self.patch_size)] += a_output[j] * self.kernel[...,:,:] #a_output[n]
                counts[x:(x+self.patch_size), y:(y+self.patch_size)] += self.kernel
        # normalize the accumulated predictions
        counts = np.repeat(np.expand_dims(counts, axis = 0), self.dataset_info.n_classes, axis = 0)
        output[counts != 0] = output[counts != 0] / counts[counts != 0] # avoid zero-division
        if criterion:
            avg_loss = np.mean(val_losses)
            return torch.tensor(np.argmax(output, axis=0)), avg_loss
        else:
            return torch.tensor(np.argmax(output, axis=0))
    
    def _generate_matrix(self, num_class, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < num_class)
        label = num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class, num_class)
        return confusion_matrix
    

    def add_batch(self, confusion_matrix, num_class, gt_image, pre_image):
        #assert gt_image.shape == pre_image.shape
        cm = confusion_matrix + self._generate_matrix(num_class, gt_image, pre_image)
        return cm

    def reset(self, num_class):
        return np.zeros((num_class,) * 2)

        
    def _get_inference(self, input_fns, target_fns, criterion = False):
        
        # evaluation (validation) 
        progress_bar = tqdm(enumerate(input_fns), total=len(input_fns))
        
        #torch.cuda.set_device()
        cm_cum=np.zeros((self.dataset_info.n_classes,) * 2)
        val_losses = []
        
        for n, fn in progress_bar:
            
            stream_data = InferenceDataset(self.dataset_info, fn, target_fns[n])
            target = stream_data.target
            if criterion:
                output, val_loss = self._infer_sample(stream_data.patches, stream_data.patch_coordinates, target, criterion)
                val_losses.append(val_loss)
            else:
                output = self._infer_sample(stream_data.patches, stream_data.patch_coordinates, target)
                     
            cm_cum += my_confusion_matrix(target.long().cpu(), output.cpu(), self.dataset_info.n_classes)
            
            if self.save_output:
                cv2.imwrite(self.save_output + '/predictions/pred_' + fn.split('/')[-1] , np.array(output)*255//(self.dataset_info.n_classes-1))
                cv2.imwrite(self.save_output + '/predictions/tar_' + fn.split('/')[-1] , np.array(target)*255//(self.dataset_info.n_classes-1))
                
        dict = cm2rates(cm_cum)
        results = rates2metrics(dict, self.dataset_info.class_names)
        
        if criterion:
            avg_loss = np.mean(val_losses)
            return results, avg_loss
        else:
            return results