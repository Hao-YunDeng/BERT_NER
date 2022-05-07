#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import pandas as pd
import numpy as np
import joblib # for saving meta data

from sklearn import preprocessing # encoders
from sklearn import model_selection # train-test split

from transformers import  AdamW #?
from transformers import get_linear_schedule_with_warmup #?

import config
import dataset
import engine
from model import EntityModel


# In[ ]:


def process_data(data_path):
    df = pd.read_csv(data_path, encoding='latin-1', on_bad_lines='skip')
    df['Sentence #'].fillna(method='ffill', inplace=True)
    
    pos_encoder = preprocessing.LabelEncoder()
    tag_encoder = preprocessing.LabelEncoder()
    
    df['POS'] = pos_encoder.fit_transform(df['POS'])
    df['Tag'] = tag_encoder.fit_transform(df['Tag'])
    
    sentences = df.groupby('Sentence #')['Word'].apply(list).values
    pos = df.groupby('Sentence #')['POS'].apply(list).values
    tag = df.groupby('Sentence #')['Tag'].apply(list).values
    
    # we need encoders as well for num of flass
    return sentences, pos, tag, pos_encoder, tag_encoder


# In[ ]:


if __name__ == '__main__':
    sentences, pos, tag, pos_encoder, tag_encoder = process_data(config.TRAINING_FILE)
    meta_data = {
        "pos_encoder": pos_encoder,
        "tag_encoder": tag_encoder
    }
    joblib.dump(meta_data, 'meta.bin')
    
    num_pos = len(list(pos_encoder.classes_))
    num_tag = len(list(tag_encoder.classes_))
    
    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1)
    
    train_dataset = dataset.EntityDataset(
        sentences=train_sentences,
        pos=train_pos,
        tag=train_tag
    )
    
    test_dataset = dataset.EntityDataset(
        sentences=test_sentences,
        pos=test_pos,
        tag=test_tag
    )    
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )    
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU available - ', device)
    else:
        device = torch.device('cpu')
        print('No GPU available, using CPU instead')
    
    model = EntityModel(num_pos, num_tag)
    model.to(device)
     
    parameters = list(model.named_parameters()) # ?
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] 
    optimizer_paras = [# ??? 
        {
            'params': [
                p for n, p in parameters if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.001
        },
        {
            'params': [
                p for n, p in parameters if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    
    optimizer = AdamW(optimizer_paras, lr=3e-5)
    num_training_steps = int(len(train_sentences) * config.EPOCHS / config.TRAIN_BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup( # ???
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train(model, train_data_loader, optimizer, device, scheduler)
        test_loss = engine.evaluation(model, test_data_loader, device)
        print(f"Train loss = {train_loss} Valid loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss


# In[ ]:




