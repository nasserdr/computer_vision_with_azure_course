import json, os, random
random.seed(1985)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random


from tqdm import tqdm
from matplotlib.patches import Rectangle
from PIL import Image


from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.autograd import Variable
from torchvision.ops import nms


from torch_snippets import *
from project_utils import *

def download_data_azure():
    '''
    Download data from an azure workspace within the confederation network.
    This function download only data from a specific RG, WS and Store
    '''
    from azureml.core import Workspace, Dataset, Datastore
    subscription_id = '78b4d5f1-fca5-4af5-b686-34747c61c20f'
    resource_group = 'computer_vision_resource_group'
    workspace_name = 'upskilling_workspace_dp'
    
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    
    datastore = Datastore.get(workspace, "workspaceblobstore")
    dataset = Dataset.File.from_files(path=(datastore, 'rumex_labelbox/images'))
    dataset.download(target_path='../../data/rumex/images', overwrite=True)


def load_clean_annotations(file, images_home):
    '''
    Returns the rumex annotations dataframe with the necessary two columns (bboxes annotations and image name).
    Specific to a labelbox dataframe with the setting we configured.

            Parameters:
                    file: full path with file name
                    images_home: where images are

            Returns:
                    annotations (pandas): pandas dataframe
    '''

    # Read the annotations cSV
    annotations = pd.read_csv(file)

    #Select the two useful columns
    annotations = annotations[['Label', 'External ID']]

    #Remove records with Label = 'Skip' or '{"bild_enthaelt_ampfer":"nein"}' or '{"bild_enthaelt_ampfer":"ja"}'
    annotations = annotations[annotations['Label'] != 'Skip']
    annotations = annotations[annotations['Label'] != '{"bild_enthaelt_ampfer":"nein"}'] #Contains no information
    annotations = annotations[annotations['Label'] != '{"bild_enthaelt_ampfer":"ja"}'] #Contains no information

    #Remove annotations for images that do not exist (or not downloaded for some reason)
    list_image = [i for i in os.listdir(images_home) if i.endswith('png') or i.endswith('jpg') or i.endswith('PNG') or i.endswith('JPG') or i.endswith('JPEG')]
    annotations = annotations[annotations['External ID'].isin(list_image)]

    #Reset the index of the dataframe
    annotations.reset_index(inplace = True, drop=True)

    #Return the dataframe
    return annotations

def get_bboxes(df, name):
    '''
    Extract bounding box for an image in a dataframe.

            Parameters:
                    df: The annotations dataframe (extracted from a labelbox annotations file)
                    name: The image name

            Returns:
                    bboxes (list): a list of bounding boxes. Each bounding boxes is a list defined by [xmin, ymin, xmax, ymax]
    '''
    row = df[df['External ID'] == name]
    #print(row)
    if len(row) == 0:
        print('The image does not have annotations (it was skipped)')
        return False
    else:
        #print(row)
        targetv1 = df['Label'].iloc[row.index[0]]
        #print(targetv1)
        targetv2 = json.loads(targetv1)
        bboxes = []
        for bbox in targetv2['Ampfer']:
            x1 = bbox['geometry'][0]['x']
            x2 = bbox['geometry'][2]['x']
            y1 = bbox['geometry'][0]['y']
            y2 = bbox['geometry'][2]['y']
            #Usually, x1 should be the xmin and so on. But, depending on how the box was drawn, they may be mixed up
            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)
            bboxes.append([xmin, ymin, xmax, ymax]) #needed in this format xmin, ymin, xmax, ymax
        return bboxes

#Define the Rumex Dataset

class RumexDataSetLabelBox(Dataset):
    def __init__(self, images_home, annotations_df, transforms = None):
        
        self.annotations_df = annotations_df
        self.root = images_home
        self.list_image = list(annotations_df['External ID'])
        #print(self.list_image)
        self.transforms = transforms
        self.datalength = len(self.list_image)
   
    def __getitem__(self, index):
        #This should return a tuple containing: the image in tensor format and the list of bounding boxes in the image
        # Given an index, we should be able to read the image from the root folder and its corresponding Bboxes
        #Bboxes should be in the following format [xmin, ymin, xmax, ymax]

        imname = self.list_image[index]
        img_path = os.path.join(self.root, imname) 
        img = Image.open(img_path).convert("RGB")

        bboxes = get_bboxes(self.annotations_df, imname)
        
        image_id = torch.tensor([index])
        #area = [ (b[3] - b[1]) * (b[2] - b[0]) for b in bboxes]

        num_rumex = len(bboxes)
        labels = torch.ones((num_rumex,), dtype=torch.int64) #Target labels should be of type int64
        target = {}
        

        target["boxes"] = torch.Tensor(bboxes)
        target["labels"] =  labels

        img = preprocess_image(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.datalength

#Reference
#https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

    #About ZIP:
    #It 'zips' tupples so that we can iterate over them in parallel 
    #(https://docs.python.org/3/library/functions.html#zip)
    #The two iterables in our case are: target and features
    #This is necessary because not all targets have the same size (same # of bboxes)
    #We give collate as an argument to the dataloader to tell DataLoader how to collate 
    #the feature/targets together.  

# define the training tranforms
def get_train_transform():
    return transforms.Compose([
        transforms.Flip(0.5),
        transforms.RandomRotate90(0.5),
        transforms.ToTensor(),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# define the validation transforms
def get_valid_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })



def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/last_model.pth')

def save_loss_plot(OUT_DIR, train_loss, val_loss):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')

    plt.close('all')

def get_model(num_classes = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights= True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array(output['labels'].cpu().detach().numpy())
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

def preprocess_image(img):
    img = np.array(img) / 255.
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float() #Float32 (avoid float 64 to not flod the memory)