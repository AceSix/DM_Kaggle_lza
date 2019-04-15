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
    
class AllConvNet2(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out


class BuildingBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 downsample=False, tweak_type='A'):
        super(BuildingBlock, self).__init__()
        # mid_channels = in_channels // 2
        stride = 2 if downsample else 1
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.residual = nn.Sequential(
                        OrderedDict([
                            ('res_avgPool', nn.AvgPool2d(2, 2)),
                            ('res_conv', nn.Conv2d(in_channels, out_channels, 1, 1)),
                            ('res_bn', nn.BatchNorm2d(out_channels))
                                    ])
                        )
        self.build_block = nn.Sequential(
                           OrderedDict([
                               ('conv1', nn.Conv2d(in_channels, out_channels, 3, stride, 1)),
                               ('bn1', nn.BatchNorm2d(out_channels)),
                               ('relu1', nn.ReLU()),
                               ('conv2', nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
                               ('bn2', nn.BatchNorm2d(out_channels))
                                       ])
                          )

    def forward(self, x):
        identity = x

        output = self.build_block(x)
        if self.downsample:
            identity = self.residual(x)
        output += identity
        output = self.relu(output)

        return output


class Resnet50(nn.Module):

    def __init__(self, stage_channels=[16, 32, 64],
                 in_channels=3, num_classes=10, tweak_type='A',
                 num_repeat=9):
        super(Resnet50, self).__init__()

        self.first_layer = nn.Sequential(
                           OrderedDict([
                               ('first_layer_conv', nn.Conv2d(in_channels, stage_channels[0], 3, 1, 1)),
                               ('first_layer_bn', nn.BatchNorm2d(stage_channels[0])),
                               ('first_layer_relu', nn.ReLU())
                                       ])
                           )

        self.stages = []
        stage_channels.insert(0, stage_channels[0])

        for i in range(len(stage_channels)-1):
            if i == 0:
                downsample = False
            else:
                downsample = True
            self.stages += self.stage_block(BuildingBlock, stage_channels[i],
                                            stage_channels[i+1], num_repeat, downsample,
                                            tweak_type)

        self.stages = nn.Sequential(*self.stages)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_channels[-1], num_classes)

    def stage_block(self, model_block, in_channels, out_channels,
                    num_repeat, downsample=True, tweak_type='A'):
        stage = [model_block(in_channels, out_channels, downsample=downsample, tweak_type=tweak_type)]
        for i in range(num_repeat - 1):
            stage += [model_block(out_channels, out_channels, tweak_type=tweak_type)]

        return stage

    def forward(self, x):
        x = self.first_layer(x)
        x = self.stages(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    


