# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:26:47 2020

@author: Kamaljeet
"""

import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
import cv2
from os.path import join as opj

class DatasetDesign(Dataset):
    def __init__(self, image_folder = 'dataset', img_width = 224, img_height = 224,
                 subset = 'train', train_per = 0.8, seed = 42):
        
        foldernames = os.listdir(image_folder)
#        foldernames = sorted(foldernames, key=lambda x: int(x.split("_")[0].split("g")[1]))
        #print('folder names sorted  ')
        
        masks_files = []
        images_files = []
        
        for folder in foldernames:
            for (root, dirs, files) in os.walk(os.path.join(image_folder , folder), topdown = True):
                for file in files:
                    if file.split('_')[0] == 'mask':
#                        masks_files.append(image_folder + folder + '/' + file)
                        masks_files.append(opj(os.path.join(image_folder, folder), file))

                    elif file.split('_')[0] == 'frame':
#                        images_files.append(image_folder + folder + '/' + file)
                        images_files.append(opj(os.path.join(image_folder, folder), file))
        print('filepaths are added')
        print(len(masks_files))
        
        #print(masks_files[0])
        
        '''
        del root
        del dirs
        del files
        del foldernames
        '''
        
        #print('images_files  ' + str(len(images_files)))
        
        images = []
        masks = []
        
        for file in masks_files:
            mask_file = resize(io.imread(file), (img_width, img_height), anti_aliasing = True)
            mask_gray = rgb2gray(mask_file)
            _,mask_file = cv2.threshold(mask_gray, 0.1, 255, cv2.THRESH_BINARY)
            mask_file = mask_file[..., np.newaxis]
            masks.append(mask_file)
        
        #print(len(masks))
        print('masks are appended')
        print('Shape of a mask before random indexing:' + str(masks[0].shape))
        
        
        for file in images_files:
            images.append(256 * resize(io.imread(file), (img_width, img_height), anti_aliasing = True))
        
        print(np.amax(images[0]))
        print('images are appended')
        print('Shape of a image before random indexing:' + str(images[0].shape))
        
        del masks_files
        del images_files
        
        '''filenames = os.listdir(image_folder)
        filenames = sorted(filenames, key=lambda x: 
            int((x.split(".")[0].split("_")[0]).split("g")[1]))
        
        self.img_width = img_width
        self.img_height = img_height
        masks = []
        images = []
        for file in filenames:
            filepath = os.path.join(image_folder, file)
            if "mask" in file:
                mask_file = imread(filepath, as_gray = True)
                mask_file = mask_file[..., np.newaxis]
                masks.append(mask_file)
            else:
                images.append(imread(filepath))'''
        
        #print('Shape of a mask before random indexing:' + str(masks[0].shape))
        
        #print(filepath)
        
        random.seed(seed)
        train_idx = random.sample(range(len(images)), int(train_per * len(images)))
        val_idx = set(range(len(images))) - set(train_idx)
                
        if subset == 'train':
            self.index = train_idx
        elif subset == 'validation':
            self.index = val_idx
        else:
            self.index = set(range(len(images)))
        
        self.image_stack = [images[i] for i in self.index]
        self.mask_stack = [masks[i] for i in self.index]
        
        print('Random Images selected')
        
        #print('Shape of mask_stack after random indexing:' + str(self.mask_stack[0].shape))
        del images
        del masks
        print(subset + ' Dataset preparation successful!!!!')
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        image = self.image_stack[idx]
        mask = self.mask_stack[idx]
        #print('From __getitem__ shape of image before crop: ' + str(image.shape))
        #print('From __getitem__ shape of mask before crop: ' + str(mask.shape))

        #image = image[:-4, :, :]
        #mask = mask[:-4, :, :]
        #print('From __getitem__ shape of image after crop: ' + str(image.shape))
        #print('From __getitem__ shape of mask after crop: ' + str(mask.shape))
        #imshow(image)

        ###################### resize #####################
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        
        
        #print('From __getitem__ shape of image after transpose: ' + str(image.shape))
        #print('From __getitem__ shape of mask after transpose: ' + str(mask.shape))
        
        #Convert to tensor form
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        #print('Shape of mask temsor: ' + str(mask_tensor.shape))
        #print(torch.max(image_tensor))
        
        return image_tensor, mask_tensor