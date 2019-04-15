import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import matplotlib as plt
from PIL import Image

def process(data, tp, transform_):
    if tp=="predict":
        img_id = data[0]
        img = data[1:3073].astype('uint8')
        img_=[]
        for i in range(0,1024):
            img_.append([img[i],img[i+1024],img[i+2048]])
        img = np.array(img_)
        img = img.reshape(32,32,3)
        img = Image.fromarray(img)
        transform = transform_
        img = transform(img)
        return img,img_id
    if tp=="train":
        target = data[3073]
        img = data[1:3073]
        img_=[]
        for i in range(0,1024):
            img_.append([img[i],img[i+1024],img[i+2048]])
        img = np.array(img_)
        img = img.reshape(32,32,3)
        img = Image.fromarray(img)
        transform = transform_
        img = transform(img)
        return img, target
        
class TrainDataset(data.Dataset):
    def __init__(self, train_data, num, transform):
        self.data = train_data
        self.num = num
        self.transform = transform
    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        img = self.data[idx]
        return process(img, "train",self.transform)
    
    
class TestDataset(data.Dataset):
    def __init__(self, test_data, num, transform):
        self.data=test_data
        self.num = num
        self.transform = transform
    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        img = self.data[idx]
        return process(img, "predict",self.transform)