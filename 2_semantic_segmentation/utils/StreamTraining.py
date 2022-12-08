
import torch
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset
from abc import ABC, abstractmethod


class TrainingDataset(IterableDataset):
    
    def __init__(self, input_fns, target_fns, dataset_info, verbose = False):
        super().__init__()
        self.fns =  list(zip(input_fns, target_fns))
        self.verbose = verbose
        self.n_fns_all = len(self.fns)
        self.dataset_info = dataset_info
        self.patch_size = dataset_info.patch_size
        self.num_patches_per_tile = dataset_info.num_patches_per_tile
        
    
    def _get_worker_range(self, fns):
        """Get the range of tiles to be assigned to the current worker"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 0
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # WARNING: when several workers are created they all have the same numpy random seed but different torch random 
        # seeds. 
        seed = torch.randint(low=0,high=2**32-1,size=(1,)).item()
        np.random.seed(seed) # set a different seed for each worker

        # define the range of files that will be processed by the current worker: each worker receives 
        # ceil(num_filenames / num_workers) filenames
        num_files_per_worker = ceil(len(fns) / num_workers)
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(len(fns), (worker_id+1) * num_files_per_worker)

        return lower_idx, upper_idx
        
    def _read_tile(self, img_fn, target_fn, rescale = False): #TO DO implement rescale if done in training
        
        #read image
        img_data = self.dataset_info.rescale(cv2.imread(img_fn))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            
        #read target
        target_data = self.dataset_info.rescale(cv2.imread(target_fn))
        
        return np.array(img_data), np.array(target_data)

    def _extract_patch(self, data, x, xstop, y, ystop):
        """
        Extract a patch from data given the sources and boundary coordinates
        """
        return data[y:ystop, x:xstop]
    
    def _get_patches_from_tile(self, *fns):

        """Generator returning patches from one tile"""
        # read data
        data = self._read_tile(*fns) #data[0] of shape (C, 2, H, W)

        if data is None:
            return #skip tile if couldn't read it

        # yield patches one by one
        for _ in range(self.num_patches_per_tile):
            data_patch, code, _ = self._generate_patch(data, None)
            if code == 1: #IndexError or invalid patch
                continue #continue to next patch
            yield data_patch
    
    def _stream_tile_fns(self, lower_idx, upper_idx):
        """Generator providing input and target paths tile by tile from lower_idx to upper_idx"""
        for idx in range(lower_idx, upper_idx):
           
            yield self.fns[idx]

    def _stream_patches(self):
        """Generator returning patches from the samples the worker calling this function is assigned to"""
        lower_idx, upper_idx = self._get_worker_range(self.fns)
        for fns in self._stream_tile_fns(lower_idx, upper_idx): #iterate over tiles assigned to the worker
            yield from self._get_patches_from_tile(*fns) #generator

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamTrainingDataset iterator")
        return iter(self._stream_patches())
    
    def _generate_patch(self, data, coord = None):
        """
        Generates a patch from the input(s) and the targets, randomly or using top left coordinates "coord"

        Args:
            - data (list of (list of) tensors): input and target data
            - num_skipped_patches (int): current number of skipped patches (will be updated)
            - coord: top left coordinates of the patch to extract, in the coarsest modality

        Output:
            - patches (list of (list of) tensors): input and target patches
            - num_skipped_patches (int)
            - exit code (int): 0 if success, 1 if IndexError or invalid patch (due to nodata)
            - (x,y) (tuple of ints): top left coordinates of the extracted patch
        """

        input_data, target_data = data

        # find the coarsest data source
        height, width = target_data.shape[:2]

        if coord is None: # pick the top left pixel of the patch randomly
            x = np.random.randint(0, width-self.patch_size)
            y = np.random.randint(0, height-self.patch_size)
            
        else: # use the provided coordinates
            x, y = coord
            
        # extract the patch
        try:
            xstop = x + self.patch_size
            ystop = y + self.patch_size
            # extract input patch
            input_patches = self._extract_patch(input_data, x, xstop, y, ystop)
            
            # extract target patch
            target_patch = self._extract_patch(target_data, x, xstop, y, ystop)
        except IndexError:
            if self.verbose:
                print("Couldn't extract patch (IndexError)")
            return (None, 1, (x, y))

        # preprocessing
        input_patches = self.dataset_info.preprocess_input(input_patches)
        target_patch = self.dataset_info.preprocess_target(target_patch)
        patches = [input_patches, target_patch]

        return (patches, 0, (x, y))
        