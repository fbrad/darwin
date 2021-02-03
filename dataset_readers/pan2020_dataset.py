from torch.utils.data import Dataset, TensorDataset
import torch
import json
import os
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from utils.data import _truncate_seq_pair 

class Pan2020Dataset(Dataset):
    def __init__(self, path: str, tokenizer: PreTrainedTokenizerBase, indexer=None):
        """
        Arguments:
            path: path to folder containing the .json files
        """
        if not os.path.exists(path):
            print("inexistent path ", path)
            raise ValueError

        self.path = path
        self.max_seq_length = 512
        # CharBERT - BasicTokenizer + Indexer
        # Others - BertTokenizer (no Indexer)
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.json_files = os.listdir(path)[:10]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise ValueError("Index out of range")
    
        json_file = os.path.join(self.path, self.json_files[idx])
        
        with open(json_file) as fp:
            entry = json.loads(fp.readline())

            if self.indexer:
                # 1) split sentence with BasicTokenizer into tokens
                # 2) further split into subtokens if possible
                # 3) index each subtoken with CharacterIndexer
                tokens_a = self.tokenizer.tokenize(entry["pair"][0])
                tokens_b = self.tokenizer.tokenize(entry["pair"][0])
                subtokens_a = []
                subtokens_b = []
                for tok in tokens_a:
                    subtokens_a.extend(self.tokenizer.tokenize(tok))
                for tok in tokens_b:
                    subtokens_b.extend(self.tokenizer.tokenize(tok))

                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "-3"
                _truncate_seq_pair(subtokens_a, subtokens_b, self.max_seq_length - 3)

                tokens = ["[CLS]"] + subtokens_a + ["[SEP]"] 
                segment_ids = [0] * len(tokens)
                tokens += subtokens_b + ["[SEP"]
                segment_ids += [1] * (len(subtokens_b) + 1)

                # convert token to ids
                # batch x List[str] = > batch x max_seq_len x 50, where 
                # max_seq_len is the same for all the dataset (512)
                input_ids = self.indexer.as_padded_tensor(
                    [tokens], 
                    maxlen=self.max_seq_length
                )[0]

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(tokens)

                # Zero-pad up to the sequence length.
                padding_length = self.max_seq_length - len(input_mask)
                input_mask += [0] * padding_length
                segment_ids += [0] * padding_length

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                labels = [1] if entry['same'] else [0]

                return {
                    "input_ids": torch.LongTensor(input_ids),
                    "token_type_ids": torch.LongTensor(segment_ids),
                    "attention_mask": torch.LongTensor(input_mask),
                    "labels": torch.LongTensor(labels),
                }
            else:
                # tokenization produces a dictionary (BaseEncoding) with "input_ids", 
                # "token_type_ids" and "attention_mask" keys
                encoding = self.tokenizer(entry["pair"][0], entry["pair"][1], truncation=True)
                encoding['labels'] = [1] if entry['same'] else [0]

                return encoding

    def __len__(self):
        return len(self.json_files)

