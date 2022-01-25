import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset
import ast 
from pathlib import Path

class CustomDataset(Dataset):
    """ATIS and SNIPS datasets for NLU."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            batch_size (callable, optional): Optional
        """
        self.csv_file = csv_file
        self.df = pd.read_pickle(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[int(idx),0 ]
        label =  self.df.iloc[int(idx),2]
        slots = self.df.iloc[int(idx),3]
        attention_mask = self.df.iloc[int(idx),1]
        token_type_ids = self.df.iloc[int(idx),4]

        return torch.tensor(text).long(),  torch.tensor(attention_mask).long(), torch.tensor(token_type_ids).long(), torch.tensor(label), torch.tensor(slots)
