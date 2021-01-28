from torch.utils.data import Dataset, TensorDataset
import json
import os
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class Pan2020Dataset(Dataset):
    def __init__(self, path: str, tokenizer: PreTrainedTokenizerBase):
        """
        Arguments:
            path: path to folder containing the .json files
        """
        if not os.path.exists(path):
            print("inexistent path ", path)
            raise ValueError

        self.path = path
        self.tokenizer = tokenizer
        self.json_files = os.listdir(path)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise ValueError("Index out of range")
    
        json_file = os.path.join(self.path, self.json_files[idx])
        with open(json_file) as fp:
            entry = json.loads(fp.readline())
            # tokenization produces a dictionary (BaseEncoding) with "input_ids", 
            # "token_type_ids" and "attention_mask" keys
            encoding = self.tokenizer(entry["pair"][0], entry["pair"][1], truncation=True)
            encoding['labels'] = [1] if entry['same'] else [0]

            return encoding

    def __len__(self):
        return len(self.json_files)

