import json
import os
import torch
from utils.character_cnn import CharacterIndexer
from torch.utils.data import Dataset, TensorDataset, dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizer

class MiniDataset(Dataset):
    """
    This dataset is built from a single pair of same-author or different
    authors documents (A, B). 
    It splits A and B into chunks of 512 tokens at most. Chunks in A
    belong to class 0, while chunks in B belong to class 1.
    """
    def __init__(self, 
                 pair_id: str,
                 doc_a: str, 
                 doc_b: str, 
                 chunk_size: int,
                 tokenizer: PreTrainedTokenizerBase,
                 indexer,
                 pair_label: int
        ):
        """
            pair_id: PAN unique id, will be used to store the pair
            doc_a: first document
            doc_b: second document
            tokenizer: Huggingface or other tokenizer with ```tokenize``` method
            pair_label: 1 (same-author) or 0 (different-authors). Will be used by
                        meta-classifier.
        """
        self.pair_id = pair_id
        self.indexer = indexer
        self.chunk_size = chunk_size
        self.pair_label = pair_label

        toks_a = tokenizer.tokenize(doc_a)
        toks_b = tokenizer.tokenize(doc_b)
        chunk_sz = chunk_size - 2 # accomodate [CLS] and [SEP]
        num_chunks_a = len(toks_a) // chunk_sz
        if len(toks_a) % chunk_sz:
            num_chunks_a += 1
        num_chunks_b = len(toks_b) // chunk_sz
        if len(toks_b) % chunk_sz:
            num_chunks_b += 1
        assert num_chunks_a > 1 and num_chunks_b > 1, 'not enough chunks'
        min_chunks = num_chunks_a if num_chunks_a < num_chunks_b else num_chunks_b
        
        # we split each document into chunks
        # we store each tokenized chunk as a single example with label 0 or 1
        fst_toks = [
            ['[CLS]'] + toks_a[idx*chunk_sz:(idx+1)*chunk_sz] + ['[SEP]'] \
            for idx in range(0, num_chunks_a)
        ]
        snd_toks = [
            ['[CLS]'] + toks_b[idx*chunk_sz:(idx+1)*chunk_sz] + ['[SEP]'] \
            for idx in range(0, num_chunks_b)
        ]
        self.input_examples = fst_toks[:min_chunks] + snd_toks[:min_chunks]
        # input examples have not been embedded yet
        self.examples = [None] * len(self.input_examples) 
        self.labels = [0] * min_chunks + [1] * min_chunks

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, idx):
        """
        Returns the numerical input required for BERT (after tokenization 
        and indexing)
        """
        tokenized_example = self.input_examples[idx]
        labels = [self.labels[idx]]

        # convert token to ids
        # batch x List[str] = > batch x max_seq_len x 50, where 
        # max_seq_len is the same for all the dataset (512)
        input_ids = self.indexer.as_padded_tensor(
            [tokenized_example], 
            maxlen=self.chunk_size
        )[0]
        padding_length = self.chunk_size - len(tokenized_example)
        segment_ids = [0] * self.chunk_size
        input_mask = [1] * len(tokenized_example) + [0] * padding_length

        return {
            "input_ids": torch.LongTensor(input_ids),
            "token_type_ids": torch.LongTensor(segment_ids),
            "attention_mask": torch.LongTensor(input_mask),
            "labels": torch.LongTensor(labels),
        }

    def save_embedding_for_example(self, idx, embedding):
        self.examples[idx] = embedding

    def get_tensor_dataset(self):
        """
        Returns the embedded examples and their labels.
        Returns a tuple with 2 Tensors of sizes (num_examples x 768, num_examples)
        """
        if None in self.examples:
            return None
        
        return (torch.stack(self.examples), torch.LongTensor(self.labels))

    def __str__(self):
        return (
            f"pair id: {self.pair_id}\n\t"
            f"label: {self.pair_label}\n\t"
            f"#examples: {len(self.input_examples)}\n\t"
            f"chunk_size: {self.chunk_size}"
        )



if __name__ == '__main__':
    dataset_path = "/pan2020/open_splits/unseen_authors/xs/pan20-av-small-test/"
    #path = "/pan2020/open_splits/unseen_authors/xs/pan20-av-small-val/aa106271-7703-5a5e-9ac7-7c280d96af9d.json"
    mini_datasets_path = os.path.join(dataset_path, "unmasking")
    if not os.path.exists(mini_datasets_path):
        os.makedirs(mini_datasets_path)

    tokenizer = BertTokenizer.from_pretrained(
        os.path.join('pretrained_models', 'bert-base-uncased'),
        do_lower_case=True
    )
    basic_tokenizer = tokenizer.basic_tokenizer
    indexer = CharacterIndexer()

    # create mini-dataset from each pair
    for idx, fname in enumerate(os.listdir(dataset_path)):
        if fname == 'unmasking':
            continue
        ex_dict = json.load(open(os.path.join(dataset_path, fname)))
        if idx % 10 == 0:
            print("%d mini-datasets created" % (idx))

        dataset = MiniDataset(
            pair_id=ex_dict['id'],
            doc_a=ex_dict['pair'][0],
            doc_b=ex_dict['pair'][1],
            chunk_size=512,
            tokenizer=basic_tokenizer,
            indexer=indexer,
            pair_label=1 if ex_dict['same'] else 0,
        )
        # save dataset
        ds_path = os.path.join(mini_datasets_path, ex_dict['id'] + ".pt")
        torch.save(dataset, ds_path)
        #break

    # load train mini-datasets

    # test_path = os.path.join(mini_datasets_path, "920a83ea-9177-5e94-8b54-4abf5868faa2.pt")
    # dataset = torch.load(test_path)
    # print(dataset)

    # print("min length: ", min(dataset_lengths))
    # print("max length: ", max(dataset_lengths))
    # print("mean length: ", sum(dataset_lengths)/len(dataset_lengths))

    # for ex in dataset.examples:
    #     print("length %d: %s" % (len(ex), ex[:5]))




        
            