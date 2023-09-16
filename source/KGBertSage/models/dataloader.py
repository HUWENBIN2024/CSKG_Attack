# Importing stock libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import sys
sys.path.append(os.getcwd())

class KGBertSage(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe
        self.head = dataframe.head_event
        self.relation = dataframe.relation
        self.tail = dataframe.tail_event
        self.label = dataframe.label
        self.is_poison = dataframe.is_poison

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'head_event': self.head[index],
            'relation': self.relation[index],
            'tail_event': self.tail[index],
            'label': self.label[index],
            'is_poison': self.is_poison[index],
        }