#%%
from torch.utils.data import Dataset
import torch
import os
import json
import random

class PAN2020(Dataset):
    def __init__(self, path, tokenizer, block_size, special_tokens=3, make_lower=False):
        assert os.path.isfile(path)

        self.tokenizer = tokenizer
        self.block_size = block_size - special_tokens
        self.data = []
        self.labels = []
        self.make_lower = make_lower
        for sample in open(path):
            sample = json.loads(sample)
            self.data.append(sample['pair'])
            self.labels.append(1 if sample['same'] == True else 0)
        self.ds_len = len(self.labels)

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        sep_token = self.tokenizer.sep_token
        cls_token = self.tokenizer.cls_token
        pad_token = self.tokenizer.pad_token_id

        if self.make_lower:
            sample1 = self.data[idx][0].lower()
            sample2 = self.data[idx][1].lower()
        else:
            sample1 = self.data[idx][0]
            sample2 = self.data[idx][1]

        sample1_tokens = self.tokenizer.tokenize(sample1)
        sample2_tokens = self.tokenizer.tokenize(sample2)

        len_s1 = len(sample1_tokens)
        len_s2 = len(sample2_tokens)

        sequence_length = int(self.block_size/2)

        if len_s1 > self.block_size:
            start_idx = random.randint(0, len_s1 - sequence_length - 1)
            sample1_tokens = sample1_tokens[start_idx: start_idx + sequence_length]
        else:
            sample1_tokens.extend([pad_token] * (sequence_length - len_s1))

        if len_s2 > self.block_size:
            start_idx = random.randint(0, len_s2 - sequence_length - 1)
            sample2_tokens = sample2_tokens[start_idx: start_idx + sequence_length]
        else:
            sample2_tokens.extend([pad_token] * (sequence_length - len_s2))

        entire_sequence = [cls_token] + sample1_tokens + [sep_token] + sample2_tokens
        attention_mask = [1 if token != pad_token else 0 for token in entire_sequence]

        token_ids = self.tokenizer.convert_tokens_to_ids(entire_sequence)

        label = self.labels[idx]

        return torch.tensor(token_ids), torch.tensor(label), torch.tensor(attention_mask)
