#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from tqdm import tqdm


# In[2]:


def train(model, data_loader, optimizer, device, scheduler):
    model.train()
    loss_avg = 0 # avg loss for a batch
    for data in tqdm(data_loader, total=len(data_loader)):
        # each data (a dict) is the info for a single sentence        
        for key, val in data.items():
            #send val to device and it becomes new val
            data[key] = val.to(device) 
            
        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_avg += loss.item()
        
    loss_avg = loss_avg / len(data_loader)
    return loss_avg


# In[3]:


def evaluation(model, data_loader, device):
    model.eval()
    loss_avg = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for key, val in data.items():
            data[key] = val.to(device)
            
        _, _, loss = model(**data)
        loss_avg += loss.item()

    loss_avg = loss_avg / len(data_loader)
    return loss_avg


# In[ ]:




