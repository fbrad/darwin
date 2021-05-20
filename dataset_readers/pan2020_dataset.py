from torch.utils.data import Dataset, TensorDataset
import torch
import json
import os
import random
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from utils.data import _truncate_seq_pair 

class Pan2020Dataset(Dataset):
    def __init__(self, path: str, tokenizer: PreTrainedTokenizerBase, 
                indexer=None, debug: bool=False, test: bool=False):
        """
        Arguments:
            path: path to folder containing the .json files (or to a .jsonl file)
        """
        if not os.path.exists(path):
            print("inexistent path ", path)
            raise ValueError

        if path.endswith(".jsonl"):
            self.folder_input = False
        else:
            self.folder_input = True

        self.path = path
        self.max_seq_length = 512
        # CharBERT - BasicTokenizer + Indexer
        # Others - BertTokenizer (no Indexer)
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.debug = debug
        self.test = test # if True, load all 512-length chunks in a document pair, not just random one
        if self.folder_input:
            self.json_files = os.listdir(path)
        else:
            self.json_files = []
            with open(path) as fp:
                for line in fp.readlines():
                    self.json_files.append(json.loads(line))

        if debug:
            self.json_files = self.json_files[:5]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise ValueError("Index out of range")

        if self.folder_input: 
            json_file = os.path.join(self.path, self.json_files[idx])
            with open(json_file) as fp:
                entry = json.loads(fp.readline())
        else:
            entry = self.json_files[idx]      

        if self.test:
            # return num_chunks sequences as a single batch element
            # 1) split sentence with BasicTokenizer into tokens
            # 2) further split into subtokens if possible
            # 3) index each subtoken with CharacterIndexer
            tokens_a = self.tokenizer.tokenize(entry["pair"][0])
            tokens_b = self.tokenizer.tokenize(entry["pair"][1])
            subtokens_a = []
            subtokens_b = []
            for tok in tokens_a:
                subtokens_a.extend(self.tokenizer.tokenize(tok))
            for tok in tokens_b:
                subtokens_b.extend(self.tokenizer.tokenize(tok))

            # shorter sequence gets padded with itself until equal to longer sequence
            len_a, len_b = len(subtokens_a), len(subtokens_b)
            max_len = len_a if len_a > len_b else len_b
            for i in range(max_len - len_a):
                subtokens_a.append(subtokens_a[i])
            for i in range(max_len - len_b):
                subtokens_b.append(subtokens_b[i])

            # get random 254 tokens from tokens_a and tokens_b
            # 508 tokens + [CLS] + [SEP] + [SEP]
            max_pair_length = self.max_seq_length // 2 - 2


            # equally spaced chunks; last unfinished chunk is shifted left
            # so that it has 254 tokens
            #print("max_len = ", max_len, ", max_pair_length = ", max_pair_length)
            lidx_list = list(range(0, max_len, max_pair_length))
            if max_len % max_pair_length:
                lidx_list[-1] = max_len-max_pair_length

            batched_input_ids = []
            batched_segment_ids = []
            batched_input_mask = []
            batched_labels = []

            for lidx in lidx_list:
                ridx = lidx + max_pair_length
                #print("(lidx, ridx) = (%d, %d)" % (lidx, ridx))
                subtokens_a_chk = subtokens_a[lidx:ridx]
                subtokens_b_chk = subtokens_b[lidx:ridx]

                tokens = ["[CLS]"] + subtokens_a_chk + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                tokens += subtokens_b_chk + ["[SEP]"]
                segment_ids += [1] * (len(subtokens_b_chk) + 1)

                #print("Tokens = ", tokens)
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

                batched_input_ids.append(input_ids)
                batched_segment_ids.append(segment_ids)
                batched_input_mask.append(input_mask)
                batched_labels.append(labels)
            
            return {
                "input_ids": torch.stack(batched_input_ids),
                "token_type_ids": torch.LongTensor(batched_segment_ids),
                "attention_mask": torch.LongTensor(batched_input_mask),
                "labels": torch.LongTensor(batched_labels),
            }
        else:
            if self.indexer:
                # 1) split sentence with BasicTokenizer into tokens
                # 2) further split into subtokens if possible
                # 3) index each subtoken with CharacterIndexer
                tokens_a = self.tokenizer.tokenize(entry["pair"][0])
                tokens_b = self.tokenizer.tokenize(entry["pair"][1])
                #print("len(tokens_a) = ", len(tokens_a))
                #print("len(tokens_b) = ", len(tokens_b))
                #print("tokens_a = ", tokens_a[0:10])
                subtokens_a = []
                subtokens_b = []
                for tok in tokens_a:
                    subtokens_a.extend(self.tokenizer.tokenize(tok))
                for tok in tokens_b:
                    subtokens_b.extend(self.tokenizer.tokenize(tok))

                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "-3"
                #_truncate_seq_pair(subtokens_a, subtokens_b, self.max_seq_length - 3)

                # get random 254 tokens from tokens_a and tokens_b
                # 508 tokens + [CLS] + [SEP] + [SEP]
                max_pair_length = self.max_seq_length // 2 - 2
                len_a, len_b = len(subtokens_a), len(subtokens_b)
                lidx_a = random.randrange(0, max(1, len_a-max_pair_length))
                ridx_a = min(lidx_a + max_pair_length, len_a)
                lidx_b = random.randrange(0, max(1, len_b-max_pair_length))
                ridx_b = min(lidx_b + max_pair_length, len_b)
                subtokens_a = subtokens_a[lidx_a:ridx_a]
                subtokens_b = subtokens_b[lidx_b:ridx_b]

                tokens = ["[CLS]"] + subtokens_a + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                tokens += subtokens_b + ["[SEP]"]
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
                encoding = self.tokenizer(
                    text=entry["pair"][0], 
                    text_pair=entry["pair"][1], 
                    truncation=True,
                    max_length=self.max_seq_length
                )
                encoding['labels'] = [1] if entry['same'] else [0]
                #print("[pan2020] encoding = ", encoding)

                return encoding

    def __len__(self):
        return len(self.json_files)

