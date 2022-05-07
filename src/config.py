#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension


# In[2]:


# from ipywidgets import FloatProgress


# In[3]:


import transformers


# In[4]:


MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "../input/bert-base-uncased"
MODEL_PATH = "model.bin" # This is where to save the model
TRAINING_FILE = "../input/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)


# In[ ]:




