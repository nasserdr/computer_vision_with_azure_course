import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from skimage.exposure import match_histograms

from scipy.interpolate import UnivariateSpline

class InferenceDataset():
    
    def __init__(self, dataset_info, img_fn, target_fn = None):
        
        img = dataset_info.rescale(cv2.imread(img_fn)) #self.rescale(Image.open(img_fn))
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if target_fn != None:
            self.tar = dataset_info.rescale(cv2.imread(target_fn)) #self.rescale(Image.open(target_fn))
        self.height, self.width = self.img.shape[:2]
        self.patch_size = dataset_info.patch_size 
        self.padding = dataset_info.padding
        self.stride = self.patch_size - self.padding
        
        self.dataset_info = dataset_info
        
        self._get_patch_coordinates()
        self.split_img()
        if target_fn != None:
            self.get_target()

    def split_img(self):

        # Setting the points for cropped image
        output = []
        for coord in self.patch_coordinates:
            x, y = coord
            img = self.img[x:x+self.patch_size, y:y+self.patch_size] #280, 470
            output.append(self.dataset_info.preprocess_input(img))
        self.patches = output
    
    def get_target(self):
        output = self.dataset_info.preprocess_target(np.array(self.tar))
        self.target = output
        
    
    def _get_patch_coordinates(self): 
        """
        Fills self.patch_coordinates with an array of dimension (n_patches, 2) containing upper left pixels of patches, 
        at the resolution of the coarsest input/targets
        """
        xs = list(range(0, self.height - self.patch_size, self.stride)) + [self.height - self.patch_size]
        ys = list(range(0, self.width - self.patch_size, self.stride)) + [self.width - self.patch_size]
        xgrid, ygrid = np.meshgrid(xs, ys)
        self.patch_coordinates = np.vstack([xgrid.ravel(), ygrid.ravel()]).T
        self._num_patches = self.patch_coordinates.shape[0]
    
    