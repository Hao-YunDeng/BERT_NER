#!/usr/bin/env python
# coding: utf-8

# In[1]:


import config
import torch
import pandas as pd


# In[2]:


class EntityDataset:
    def __init__(self, sentences, pos, tag):
        self.sentences = sentences
        self.pos = pos
        self.tag = tag
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        pos = self.pos[index]
        tag = self.tag[index]
        
        ids = []
        target_pos = []
        target_tag = []
        
        # Tokenizing the sentence
        for i, word in enumerate(sentence):
            token = config.TOKENIZER.encode(
                word,
                add_special_tokens=False
            )
            token_len = len(token)
            ids.extend(token)
            target_pos.extend([pos[i]] * token_len)
            target_tag.extend([tag[i]] * token_len)
        # Now, ids list is tokens for words in the sentence  
        # each element in ids list is a token for word/piece of a word
        
        # Padding
        ids = ids[:config.MAX_LEN - 2]
        target_pos = target_pos[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]
        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids) # the valid words have mask=1, while paddings have mask=0
        token_type_ids = [0] * len(ids) # used for paired sentences

        padding_len = config.MAX_LEN - len(ids)
        ids = ids + [0] * padding_len
        target_pos = target_pos + [0] * padding_len
        target_tag = target_tag + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        
        # return the info of a sentence in a dictionary
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),           
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
        }


# In[3]:


if __name__ == '__main__':
    ner_dataset_df = pd.read_csv("../input/ner_dataset.csv", encoding = "latin-1")
    display(ner_dataset_df)
    
    ner_df = pd.read_csv("../input/ner.csv", encoding = "latin-1", on_bad_lines='skip')
    display(ner_df)


# In[ ]:




