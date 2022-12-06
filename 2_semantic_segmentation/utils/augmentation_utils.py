import imageio
from PIL import Image
from glob import glob
import os
import cv2
from tqdm import tqdm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

# Import matplotlib for visualization
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class AugmentData():
    
    def __init__(self, root, csv_file, img_dir, msk_dir, out_dir):
        root = root
        csv_file = csv_file
        self.filenames = [p[0] for p in csv_file]
        self.img_dir = img_dir
        self.mask_dir = msk_dir
        

    def applyDataAug(self, n):
        
        for f in tqdm(self.filenames):
            
            print('Processing image ', f,' ...')
        
            imgPath = glob(self.img_dir + '/cropped_' + f + '*')[0]
            maskPath = glob(self.mask_dir + '/' + f + '*')[0]
            
            image, segmap = self.getImMk(imgPath, maskPath)
            augmented = self.augment(image, segmap, n)
            
            self.saveResults(augmented, f)
    
    def augment(self, image, segmap, n):
        
        sometimes = lambda x: iaa.Sometimes(0.5, x)
        seq = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5), # horizontally flip 50% of all images
                    iaa.Flipud(0.5), # vertically flip 50% of all images
                    
                    sometimes(iaa.Affine(
                        rotate=(-45,45), # rotate by -45 to 45 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    )),

                    sometimes(iaa.Affine(scale=(0.5, 1.5))), #rescale from 50% (zoom in) ton 150% (zoom out)

                    # execute 0 to 3 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 3),
                        [
                            iaa.GaussianBlur((0, 1.3)), # blur images with a sigma between 0 and 3.0
                            iaa.WithBrightnessChannels(iaa.Add((-70, 50))), # change brightness of images (by -10 to 10 of original value)          
                            iaa.AdditiveGaussianNoise(scale=(0.05*255, 0.1*255)), #add Gaussian noise
                            # either change the brightness of the whole image (sometimes
                            
                            # per channel) or change the brightness of subareas
                            iaa.LinearContrast((0.8, 1.1)), # improve or worsen the contrast
                        ],
                        random_order=False
                    )
                ],
                random_order=False
        )
        augmented = [seq(image=image, segmentation_maps=segmap) for _ in range(n)]
        return augmented
    
    def saveResults(self, augmented, fn):
        for i in np.arange(len(augmented)):
            img_res = Image.fromarray(augmented[i][0].astype('uint8'), 'RGB')
            mask_res = Image.fromarray(np.array(augmented[i][1].get_arr()).astype('uint8')*255, 'RGB')
            
            xmin, ymin, xmax, ymax= self.findEdge(img_res)
            img_cropped = self.cropImg(img_res, xmin, ymin, xmax, ymax)
            mask_cropped = self.cropImg(mask_res, xmin, ymin, xmax, ymax)

            img_cropped.save(self.img_dir + "/cropped_" + fn + "_A" + str(i) + ".jpg")
            mask_cropped.save(self.mask_dir + "/" + fn +"_A" + str(i) + ".jpg")
        print('... Saved.')
    
    def findEdge(self, img):
        
        arr = np.array(img)
        cond = np.logical_and(arr[:,:,0] == arr[:,:,1], arr[:,:,0] == arr[:,:,2])
        X, Y = np.nonzero(np.invert(cond))
        xmin, ymin = X[0], Y[0]
        xmax, ymax = X[-1], Y[-1]
        return xmin, ymin, xmax, ymax
                
    def cropImg(self, img, xmin, ymin, xmax, ymax):
        # Size of the image in pixels (size of original image)
        # (This is not mandatory)
        
        # Setting the points for cropped image
        left = ymin
        top = xmin
        right = ymax
        bottom = xmax
        
        # Cropped image of above dimension
        # (It will not change original image)
        im1 = img.crop((left, top, right, bottom))
        
        return im1
    
        
    def getImMk(self, imgPath, mskPath):
    
        # use imageio library to read the image (alternatively you can use OpenCV cv2.imread() function)
        image = imageio.imread(imgPath)

        # Open image with mask
        pil_mask = Image.open(mskPath)
        pil_mask = pil_mask.convert("RGB")

        # Convert mask to binary map
        np_mask = np.array(pil_mask)
        np_mask[np_mask > 127] = 255
        np_mask[np_mask <= 127] = 0
        np_mask = np.clip(np_mask, 0, 1)

        # Create segmentation map
        segmap = np.zeros(image.shape, dtype=bool)
        segmap[:] = np_mask
        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
        
        return image, segmap

def plotAugmentation(img_paths):

    img_list = []

    for p in img_paths:
        img = cv2.imread(p)
        img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # create figure
    fig = plt.figure(figsize=(16, 7))

    # setting values to rows and column variables
    rows = 2
    columns = 3
    title = ['Original', 'Augmented 1', 'Augmented 2', 'Augmented 3', 
    'Augmented 4', 'Augmented 5']

    for i in range(6):
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i + 1)
        # showing image
        plt.imshow(img_list[i])
        plt.axis('off')
        plt.title(title[i])
    
if __name__ == "__main__":
        
    aug_data = AugmentData()
    
    aug = aug_data.applyDataAug(5)