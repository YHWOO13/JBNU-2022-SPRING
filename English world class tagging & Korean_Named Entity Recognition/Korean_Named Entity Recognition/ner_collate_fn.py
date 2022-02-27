import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

def make_random_len_data_list(min_len, max_len, num_data):
    random_data = []
    
    for i in range(num_data):
        sample_len = random.randrange(min_len, max_len)
        sample = [random.randint(0, 9) for ii in range(sample_len)]
        random_data.append(sample)
    
    return random_data

class Dataset_custom(Dataset):
    def __init__(self, data):
        self.x = data
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]

def make_same_len(batch):
    
    each_len_list = [len(sample) for sample in batch]
    print('each_len_list', each_len_list)
    
    max_len = max(each_len_list)
    
    padded_batch = []
    pad_id = 0
    
    for sample in batch:
        padded_batch.append(sample + [pad_id] * (max_len - len(sample)))
    
    return padded_batch

def collate_fn_custom(batch):
    padded_batch = make_same_len(batch)
    
    padded_batch = torch.tensor(padded_batch)
    
    return padded_batch