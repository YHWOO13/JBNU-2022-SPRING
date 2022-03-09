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
    max_len = max(each_len_list)
    
    padded_batch = []
    pad_id = 0
    special_token = 0    

    for sample in batch:
        padded_batch.append([special_token] + sample + [pad_id] * (max_len - len(sample)) + [special_token])
    
    return padded_batch

def collate_fn_custom(data):
    inputs = [sample[0] for sample in data]
    labels = [sample[2] for sample in data]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first = True)
    
    return {'input': padded_inputs.contiguous(),
            'label': torch.stack(labels).contiguous()}