# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:53:21 2022

@author: Admin
"""

import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from utils_camvid import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random

def augmentation():
    pass

def augmentation_pixel():
    pass

class CamVid(torch.utils.data.Dataset):
    
    def __init__(self,image_path,label_path,csv_path,scale,loss='dice',mode='train'):
        super().__init__()
        self.mode=mode
        self.img_list=[]
        if not isinstance(image_path,list):
            image_path=[image_path]
        for img in image_path:
            self.img_list.extend(glob.glob(os.path.join(img,'*.png')))
        self.img_list.sort()
        self.label_list=[]
        if not isinstance(label_path,list):
            label_path=[label_path]
        for label in label_path:
            self.label_list.extend(glob.glob(os.path.join(label,'*.png')))
        self.label_list.sort()
        self.fliplr=iaa.Fliplr(0.5)
        self.label_info=get_label_info(csv_path)
        self.to_tensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
            ])
        self.img_size=scale
        self.scale=[0.5,1,1.25,1.5,1.75,2]
        self.loss=loss
    
    def __getitem__(self,idx):
        
        seed=random.random()
        img=Image.open(self.img_list[idx])
        scale=random.choice(self.scale)
        scale=(int(self.img_size[0]*scale),int(self.img_size[1]*scale))
        #resize and crop
        if self.mode=='train':
            img=transforms.Resize(scale,Image.BILINEAR)(img)
            img=RandomCrop(self.img_size,seed,pad_if_needed=True)(img)
        img=np.array(img)
        #load label
        label=Image.open(self.label_list[idx])
        #resize and crop label
        if self.mode=='train':
            label=transforms.Resize(scale,Image.NEAREST)(label)
            label=RandomCrop(self.img_size,seed,pad_if_needed=True)(label)
        label=np.array(label)
        #Augment img and label
        if self.mode=='train':
            seq_det=self.fliplr.to_deterministic()
            img=seq_det.augment_image(img)
            label=seq_det.augment_image(label)
        img=Image.fromarray(img)
        img=self.to_tensor(img).float()
        
        if self.loss=='dice':
            label=one_hot_it_v11_dice(label,self.label_info).astype(np.uint8)
            label=np.transpose(label,[2,0,1]).astype(np.float32)
            label=torch.from_numpy(label)
        elif self.loss=='crossentropy':
            label=one_hot_it_v11(label,self.label_info).astype(np.uint8)
            label=torch.from_numpy(label).long()
        return img,label
    def __len__(self):
        return len(self.img_list)

if __name__=='__main__':
    data=CamVid(['CamVid/train','CamVid/val'],['CamVid/train_labels','CamVid/val_labels'],
                'CamVid/class_dict.csv',(720,960),loss='crossentropy',mode='val')
    label_info=get_label_info('CamVid/class_dict.csv')
    for i,(img,label) in enumerate(data):
        print(label.size())
        print(torch.max(label))

    
        
