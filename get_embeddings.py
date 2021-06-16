from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizerFast, BertTokenizer, TrainingArguments, Trainer, BertConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSequenceClassification
from transformers.models.bert import BertForSequenceClassification
from transformers import BertModel
from transformers.optimization import get_constant_schedule
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import SGD
from models.character_bert import CharacterBertModel
from models.character_bert_for_classification import CharacterBertForSequenceClassification
from dataset_readers.pan2020_dataset import Pan2020Dataset
from dataset_readers.mini_dataset import MiniDataset
from dataset_readers.pan2020_segments_dataset import Pan2020SegmentsDataset
from utils.training import train, evaluate, PanTrainer, LogCallback
from utils.metrics import evaluate_all, compute_pan_metrics
from utils.character_cnn import CharacterIndexer
from utils.misc import set_seed, parse_args
import numpy as np
import pickle
import sys
import math
import torch
import logging
import os
from tqdm import tqdm
from typing import Dict
from transformers.trainer_utils import EvalPrediction
from transformers.integrations import TensorBoardCallback
import json
import sys


if __name__ == '__main__':   
    # read config file and merge with arguments
    args = parse_args()

    # set up logging
    # TODO: modify this to another checkpoint folder
    #args.output_dir = "/data/darwin/xs_open_uf_2e-5/checkpoint-13818"
    args.output_dir = "/data/darwin/xs_open_ua_2e-5/checkpoint-8430"

    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'output_test.log'),
        format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        filemode='w',
        force=True
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.info('Arguments: %s ', args)

    # set up tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join('pretrained_models', 'bert-base-uncased'),
        do_lower_case=args.do_lower_case
    )
    tokenizer = tokenizer.basic_tokenizer
    characters_indexer = CharacterIndexer() 

    # set up test dataset and dataloader
    # TODO: modify this to another test dataset
    # args.test_path = "/pan2020/closed_splits/closed_split_v1/xs/pan20-av-small-test"
    #args.test_path = "/pan2020/open_splits/unseen_fandoms/xs/pan20-av-small-test"
    args.mini_datasets_path = "/pan2020/open_splits/unseen_authors/xs/pan20-av-small-test/unmasking"
    mini_datasets = []
    for idx, dataset_fname in enumerate(os.listdir(args.mini_datasets_path)):
        mini_datasets.append(torch.load(
            os.path.join(args.mini_datasets_path, dataset_fname)
        ))
    
    # set up config
    config = BertConfig.from_pretrained(
        os.path.join('pretrained_models', "general_character_bert"),
        num_labels=2
    )
    config.update({"return_dict": False})

    # set up backbone
    logging.info('Loading classification model with %s backbone', "general_character_bert")

    model = BertForSequenceClassification(config=config)

    # this backbone is overwritten by fine-tuned backbone below
    model.bert = CharacterBertModel.from_pretrained(
       os.path.join('pretrained_models', "general_character_bert"),
       config=config
    )

    # load pretrained weights
    print("Args.output_dir before state_dict = ", args.output_dir)
    state_dict = torch.load(
        os.path.join(args.output_dir, 'pytorch_model.bin'), map_location='cpu'
    )
    model.load_state_dict(state_dict, strict=True)
    #args.device = 'cpu'
    model.to(args.device)
    logging.info("model = %s" % (model))

    # get embeddings for all mini-datasets
    model.eval()
    for idx, mini_dataset in enumerate(mini_datasets):
        print("processed %d mini-datasets" % (idx))
        sampler = SequentialSampler(mini_dataset)
        dataloader = DataLoader(
            mini_dataset,
            sampler=sampler,
            batch_size=len(mini_dataset)
        )

        for batch in dataloader:
            batch = {k: v.to(device=args.device) for k, v in batch.items()}
            batch['return_dict'] = False
            del batch['labels']

            with torch.no_grad():
                # (batch_size x 512 x 768, batch_size x 768)
                outputs = model.bert(**batch)
                dataset_embeddings = outputs[1]
                dataset_size = len(mini_dataset)
                # save embeddings in mini-dataset
                for idx in range(dataset_size):
                    mini_dataset.save_embedding_for_example(
                        idx, 
                        dataset_embeddings[idx]
                    )

                # save updated dataset
                torch.save(
                    mini_dataset,
                    os.path.join(
                        args.mini_datasets_path, 
                        mini_dataset.pair_id + ".pt"
                    )
                )

    # ds_lengths = []
    # for idx, mini_ds in enumerate(mini_datasets):
    #     print("Dataset %d" % (idx))
    #     ds_lengths.append(len(mini_ds))
    #     ds_tensor = mini_ds.get_tensor_dataset()
    #     print("embedded examples size: ", ds_tensor[0].size())
    #     print("labels: ", ds_tensor[1].size())
    #     #print("  length = ", len(mini_ds), " size = ", ds_tensor[0].size())