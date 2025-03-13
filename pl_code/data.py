import os
from torch.utils.data import Dataset, DataLoader, Subset, SequentialSampler
import pandas as pd
import pickle as pkl
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler



class MyDataset(Dataset):
    def __init__(self, input_dict, label):
        self.input_dict = input_dict
        self.label = label 


    def __getitem__(self, item):
        sample_dict = {}
        for key in self.input_dict.keys():
            sample_dict[key] = self.input_dict[key][item]
        sample_dict["label"] = self.label[item]

        return sample_dict

    def __len__(self):
        return len(self.input_dict['101'])
    

