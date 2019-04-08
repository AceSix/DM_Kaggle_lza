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

def process(data, tp):
    if tp=="predict":
        img_id = data[0]
        img = data[1:3073]
        img = img.reshape(3,32,32)
        img = img/256
        img = (img-0.5)/0.5
        return img,img_id
    if tp=="train":
        target = data[3073]
        img = data[1:3073]
        img = img.reshape(3,32,32)
        img = img/256
        img = (img-0.5)/0.5
        return img, target
        
class TrainDataset(data.Dataset):
    def __init__(self, train_data):
        self.data = train_data
    def __len__(self):
        return 40000
    def __getitem__(self, idx):
        img = self.data[idx]
        return process(img, "train")
    
class ValidDataset(data.Dataset):
    def __init__(self, valid_data, valid_num):
        self.data=valid_data
        self.num = valid_num
    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        img = self.data[idx]
        return process(img, "train")
    
class TestDataset(data.Dataset):
    def __init__(self, test_data):
        self.data=test_data
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        img = self.data[idx]
        return process(img, "predict")