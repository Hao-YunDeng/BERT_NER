#!/usr/bin/env python
# coding: utf-8

# In[1]:


import config
import torch
import transformers
import torch.nn as nn


# In[2]:


def criterion(output, target, mask, num_of_labels):
    cross_entropy_loss = nn.CrossEntropyLoss()    
    output = output.view(-1, num_of_labels)
    mask = mask.view(-1) == 1   
    
    active_target = torch.where(
        mask,
        target.view(-1),
        torch.tensor(cross_entropy_loss.ignore_index).type_as(target)        
    )
    loss = cross_entropy_loss(output, active_target)
    return loss


# In[3]:


class EntityModel(nn.Module):
    def __init__(self, num_of_pos, num_of_tag):
        super(EntityModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL_PATH, 
            return_dict=False #important to make sure the output format
        )
        self.num_of_pos = num_of_pos
        self.num_of_tag = num_of_tag
        
        self.drop_for_pos = nn.Dropout(p=0.3)
        self.drop_for_tag = nn.Dropout(p=0.3)
        
        self.linear_for_pos = nn.Linear(768, self.num_of_pos)
        self.linear_for_tag = nn.Linear(768, self.num_of_tag)
        
        
        
    def forward(self, ids, target_pos, target_tag, mask, token_type_ids):
        hidden_state, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        pos = self.drop_for_pos(hidden_state)
        pos = self.linear_for_pos(pos)
        
        tag = self.drop_for_tag(hidden_state)
        tag = self.linear_for_tag(tag)
        
        loss_for_pos = criterion(pos, target_pos, mask, self.num_of_pos)
        loss_for_tag = criterion(tag, target_tag, mask, self.num_of_tag)
        
        loss = (loss_for_pos + loss_for_tag) / 2
        return pos, tag, loss


# In[ ]:




