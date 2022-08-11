# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

from torch.utils.data import Dataset, DataLoader
from config import *
from utils import *
import numpy as np
import torch
import cv2
import os

class ClarifyNetDataLoader(Dataset):
    def __init__(self, img_paths = None, mask_paths = None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        assert len(self.img_paths) == len(self.mask_paths)
        self.images = len(self.img_paths) # list all the files present in images folder...
        
    def __len__(self):
        return len(self.img_paths) # return length of dataset
        
    def lowPass(self, img):
        temp = img.copy()
        dest = cv2.GaussianBlur(temp, (5, 5), cv2.BORDER_DEFAULT) 
        return dest
    
    def highPass(self, img):
        temp = img.copy()
        dest = cv2.GaussianBlur(temp, (3, 3), 0) 
        source_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
        dest = cv2.Laplacian(source_gray, cv2.CV_16S, ksize = 3)
        abs_dest = cv2.convertScaleAbs(dest)
        return abs_dest
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        
        if COLOR_FORMAT == "YCrCb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2YCrCb)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        lowpass = self.lowPass(mask)
        highpass = self.highPass(mask)
        
        image = image.astype(np.float32)
        image = image/255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        
        lowpass = lowpass.astype(np.float32)
        lowpass = lowpass/255.0
        lowpass = torch.from_numpy(lowpass)
        lowpass = lowpass.permute(2,0,1)
        
        highpass = highpass.astype(np.float32)
        highpass = highpass[:,:,np.newaxis]
        highpass = highpass/255.0
        highpass = torch.from_numpy(highpass)
        highpass = highpass.permute(2,0,1)
        
        mask = mask.astype(np.float32)
        mask = mask/255.0
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        
        return image, lowpass, highpass, mask

def getDataLoader(batch_size):
    X_train = sorted([os.path.join(TRAIN_IMAGES_DIR, img) for img in os.listdir(TRAIN_IMAGES_DIR)])
    y_train = sorted([os.path.join(TRAIN_LABELS_DIR, lbl) for lbl in os.listdir(TRAIN_LABELS_DIR)])

    X_val = sorted([os.path.join(VAL_IMAGES_DIR, img) for img in os.listdir(VAL_IMAGES_DIR)])
    y_val = sorted([os.path.join(VAL_LABELS_DIR, lbl) for lbl in os.listdir(VAL_LABELS_DIR)])

    X_test = sorted([os.path.join(TEST_IMAGES_DIR, img) for img in os.listdir(TEST_IMAGES_DIR)])
    y_test = sorted([os.path.join(TEST_LABELS_DIR, lbl) for lbl in os.listdir(TEST_LABELS_DIR)])

    train_ds = ClarifyNetDataLoader(img_paths = X_train, mask_paths = y_train)
    train_loader = DataLoader(
        train_ds, 
        batch_size = batch_size, 
        num_workers = NUM_WORKERS,
        shuffle = SHUFFLE
        )
    val_ds = ClarifyNetDataLoader(img_paths = X_val, mask_paths = y_test)
    val_loader = DataLoader(
        val_ds, 
        batch_size = batch_size, 
        num_workers = NUM_WORKERS, 
        shuffle = False
        )
    test_ds = ClarifyNetDataLoader(img_paths = X_test, mask_paths = y_test)
    test_loader = DataLoader(
        test_ds, 
        batch_size = batch_size, 
        num_workers = NUM_WORKERS, 
        shuffle = False
        )
    
    return train_loader, val_loader, test_loader