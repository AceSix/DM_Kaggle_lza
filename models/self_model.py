# encoding:utf-8
# Modify from torchvision
# ResNeXt: Copy from https://github.com/last-one/tools/blob/master/pytorch/SE-ResNeXt/SeResNeXt.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import math
from collections import OrderedDict


nn.MaxPool2d(kernel_size=3, stride=2, padding=1,ceil_mode=False)


class SimpleNet(nn.Module):
    def __init__(self, N_hid = 3000):    
        super(SimpleNet, self).__init__()  
        self.sub_layer = self.Subsample()
        self.conv_mid = nn.Conv2d(192, 192, 1, 1, 0)
        self.drop_layer1 = self.Dropout_Layer(12288, 1000, N_hid)
        self.fc_end = nn.Linear(1000, 10)
                
    def Dropout_Layer(self, N_pass,N_out, N_HIDDEN):
        dropout_layer = torch.nn.Sequential(
            nn.Linear(N_pass, N_HIDDEN),
            nn.Dropout(0.5),           
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_out),
            nn.ReLU()
        )
        return dropout_layer
            
    def Subsample(self):
        subsample = torch.nn.Sequential(
            nn.Conv2d(3, 96, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1,ceil_mode=False),
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, 1, 1),
#             nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1,ceil_mode=False),
            nn.ReLU(),
            
        )
        return subsample

    def forward(self, x):                  
        x = self.sub_layer(x)
#         x = F.relu(self.conv_mid(x))
        x = x.view(x.size(0), -1)
        x = self.drop_layer1(x)
        x = self.fc_end(x)
        return x


class AllConvNet(nn.Module):
    def __init__(self, N_hid):    
        super(AllConvNet, self).__init__()  
        self.sub_layer = self.Subsample()
        self.conv_mid = nn.Conv2d(192, 192, 1, 1, 0)
        self.drop_layer1 = self.Dropout_Layer(12288, 3000, N_hid)
        self.fc_end = nn.Linear(12288, 10)
                
    def Dropout_Layer(self, N_pass,N_out, N_HIDDEN):
        dropout_layer = torch.nn.Sequential(
            nn.Linear(N_pass, N_HIDDEN),
            nn.Dropout(0.5),           
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_out),
            nn.ReLU()
        )
        return dropout_layer
            
    def Subsample(self):
        subsample = torch.nn.Sequential(
            nn.Conv2d(3, 96, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(),
           
            nn.Conv2d(96, 96, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, 1, 1),
#             nn.BatchNorm2d(192),
            nn.ReLU(),
            
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, 2, 1),
            nn.ReLU(),
            
        )
        return subsample
            
    def forward(self, x):                  
        x = self.sub_layer(x)
        x = F.relu(self.conv_mid(x))
        x = x.view(x.size(0), -1)
#         x = self.drop_layer1(x)
        x = self.fc_end(x)
        return x
    
