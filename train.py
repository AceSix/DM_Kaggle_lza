
# coding: utf-8

# In[3]:


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
import matplotlib.pyplot as plt
import math
import self_model as models
import get_data


# In[2]:


data_path = "C://Users//shjdl//Desktop//DM_project//data//"
train_path = data_path+"train1.csv"
valid_path = data_path+"validation.csv"

raw_data = pd.read_csv(train_path, iterator=True)
all_data = raw_data.get_chunk(50000).values.astype('uint8')
train_data = all_data[0:45000]
valid_data = all_data[45000:50000]
all_data = None


# In[5]:


BATCH_SIZE = 128
NUM_EPOCHS = 300
LR = 0.1

# model = models.SEResNeXt(models.BottleneckX, [3, 4, 6, 3],cardinality=32, num_classes=10)
# model = models.AllConvNet(3000)
model = models.Resnet50()
criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adadelta(model.parameters(), 0.9)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 150, gamma = 0.1)

# optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))

train_dataset = get_data.TrainDataset(train_data, 45000)
valid_dataset = get_data.ValidDataset(valid_data,500)
test_dataset = get_data.ValidDataset(train_data,500)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# In[6]:


tloss = []
epochloss=[]
valid = []
test = []
processed = 0

for epoch in range(NUM_EPOCHS):    
    
    scheduler.step()
    
    for images, labels in tqdm(train_loader,disable = False):
        images = images.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        model = model.cuda()
        
        out = model(images)

        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

            
    print("epoch %d,current entropy is %f" % (epoch, loss.data.item()))
    tloss.append(loss.data.item())        
    train_acc = 0
    test_acc = 0
    
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for images, labels in tqdm(test_loader,disable =True):
        images = images.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)
        images = images.cuda()
        labels = labels.cuda()
        model = model.cuda()
        out = model(images)
        _, pred = torch.max(out, 1)
        
        num_correct = (pred == labels).sum()
        test_acc += num_correct.item()
    test.append(test_acc / 500)  
    
    valid_loader = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for images, labels in tqdm(valid_loader,disable =True):
        images = images.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)
        images = images.cuda()
        labels = labels.cuda()
        model = model.cuda()
        out = model(images)
        _, pred = torch.max(out, 1)
        
        num_correct = (pred == labels).sum()
        train_acc += num_correct.item()
    epochloss.append(loss.data.item())
    valid.append(train_acc / 500)
    
    print("current test accuracy is %f , current valid accuracy is %f " % (test_acc/500, train_acc / 500))


# In[ ]:


datetime =time.strftime('%Y.%m.%d',time.localtime(time.time()))
PATH = "./model-" +datetime+ ".pt"
torch.save(model, PATH) 

model=torch.load(PATH)
plt.plot(epochloss)
plt.plot(valid)
plt.plot(test)
plt.show()

