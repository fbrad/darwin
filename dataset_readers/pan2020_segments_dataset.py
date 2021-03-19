from torch.utils.data import Dataset, TensorDataset
import torch
import json
import os
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from utils.data import _truncate_seq_pair 

class Pan2020SegmentsDataset(Dataset):
    """
    Splits example pairs (x, y) into several chunks of at most 512 tokens, resulting
    in a sequence tuple (x1, x2, ..., xn, y1, y2, ..., ym). Each sequence xi/yi
    is
    """

    def __init__(self, path: str, tokenizer: PreTrainedTokenizerBase, 
                indexer=None, debug: bool=False):
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
        self.debug = debug
        self.json_files = os.listdir(path)[:10] if debug else os.listdir(path)

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
                tokens_b = self.tokenizer.tokenize(entry["pair"][1])
                subtokens_a = []
                subtokens_b = []
                for tok in tokens_a:
                    subtokens_a.extend(self.tokenizer.tokenize(tok))
                for tok in tokens_b:
                    subtokens_b.extend(self.tokenizer.tokenize(tok))
                
                doc_input_ids = [] # num_chunks x max_seq_length x 50
                doc_input_mask = [] # num_chunks x max_seq_length
                doc_segment_ids = [] # num_chunks x max_seq_length
                # split subtokens_a into chunks of at most 510 tokens
                chunk_max_len = self.max_seq_length - 2 
                for subtokens in [subtokens_a, subtokens_b]:
                    num_chunks = len(subtokens) // chunk_max_len
                    if len(subtokens_a) % chunk_max_len > 0:
                        num_chunks += 1
                    for chunk_lidx in range(0, len(subtokens), self.max_seq_length-2):
                        chunk_ridx = min(chunk_lidx + self.max_seq_length - 2, len(subtokens))
                        chunk_len = chunk_ridx - chunk_lidx
                        #print("[%d, %d], len = %d" % (chunk_lidx, chunk_ridx, chunk_len))
                        chunk_tokens = ['[CLS]'] + subtokens[chunk_lidx:chunk_ridx] + ['[SEP]']

                        padding_length = self.max_seq_length - len(chunk_tokens)
                        chunk_input_mask =  [1] * len(chunk_tokens) + [0] * padding_length
                        chunk_segment_ids = [0] * self.max_seq_length
                        #print("len(chunk_input_mask) = %d, len(chunk_segment_ids) = %d" % (len(chunk_input_mask), len(chunk_segment_ids)))

                        # convert token to ids
                        # max_seq_len = > max_seq_len x 50
                        chunk_input_ids = self.indexer.as_padded_tensor(
                            [chunk_tokens], 
                            maxlen=self.max_seq_length
                        )[0]
                        #print("chunk_input_ids size = ", chunk_input_ids.size())

                        doc_input_ids.append(chunk_input_ids)
                        doc_input_mask.append(chunk_input_mask)
                        doc_segment_ids.append(chunk_segment_ids)
               
                # assert len(doc_input_ids) == num_chunks
                # assert len(doc_input_mask) == num_chunks
                # assert len(doc_segment_ids) == num_chunks

                # num_chunks x 512 x 50
                doc_input_ids = torch.stack(doc_input_ids)
                # num_chunks x 512
                doc_input_mask = torch.LongTensor(doc_input_mask)
                # num_chunks x 512
                doc_segment_ids = torch.LongTensor(doc_segment_ids)

                labels = [1] if entry['same'] else [0]

                return {
                    "input_ids": doc_input_ids,
                    "token_type_ids": doc_segment_ids,
                    "attention_mask": doc_input_mask,
                    "num_chunks": torch.LongTensor([len(doc_input_ids)]),
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

    def collate_fn(examples):
        max_len = max([x['input_ids'].size(0) for x in examples])
        #print("max_len = ", max_len)
        out_dict = {}
        for example in examples:
            padding_length = max_len - example['input_ids'].size(0)
            # padding_length x 50
            if padding_length > 0:
                padding_input_tensor = example['input_ids'].new_full(
                    (padding_length, 512, 50),
                    0
                )
                padding_mask_tensor = example['token_type_ids'].new_full(
                    (padding_length, 512),
                    0
                )
                #print("Padding tensor size = ", padding_tensor.size())
                example['input_ids'] = torch.cat((example['input_ids'], padding_input_tensor))
                example['token_type_ids'] = torch.cat((example['token_type_ids'], padding_mask_tensor))
                example['attention_mask'] = torch.cat((example['attention_mask'], padding_mask_tensor))
            #print("input size = ", ex['input_ids'].size())]
        #return tokenizer.pad(examples, padding=True, return_tensors='pt')
        out_dict['input_ids'] = torch.stack([ex['input_ids'] for ex in examples])
        out_dict['token_type_ids'] = torch.stack([ex['token_type_ids'] for ex in examples])
        out_dict['attention_mask'] = torch.stack([ex['attention_mask'] for ex in examples])
        out_dict['num_chunks'] = torch.stack([ex['num_chunks'] for ex in examples])
        out_dict['labels'] = torch.stack([ex['labels'] for ex in examples])
        #for k, v in out_dict.items():
        #    print("%s size = %s" % (k, v.size()))

        return out_dict