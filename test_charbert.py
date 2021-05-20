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
    args.output_dir = "/data/darwin/xs_open_uf_2e-5/checkpoint-13818"
    #args.output_dir = "/data/darwin/xs_open_ua_2e-5/checkpoint-8430"

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
    args.test_path = "/pan2020/open_splits/unseen_fandoms/xs/pan20-av-small-test.jsonl"
    test_dataset = Pan2020Dataset(
        args.test_path, 
        tokenizer=tokenizer, 
        indexer=characters_indexer, 
        debug=False,
        test=True
    )
    # test_sampler = SequentialSampler(eval_dataset)
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     sampler=test_sampler,
    #     batch_size=args.eval_batch_size
    # )

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

    # train model using custom train/eval loop
    # global_step, train_loss, best_val_metric, best_val_epoch = train(
    #     args=args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     model=model
    # )
    #logging.info("global_step = %s, average training loss = %s", global_step, train_loss)
    #logging.info("Best performance: Epoch=%d, Value=%s", best_val_epoch, best_val_metric)
    
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        disable_tqdm=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_dir=args.output_dir,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='overall',
        greater_is_better=True
    )
        
    args.eval_batch_size = 1
    # unpickle results
    # results, preds_list, probs_list, out_label_ids = evaluate(
    #     args, 
    #     test_dataset, 
    #     model, 
    #     test_mode=True
    # )
    # print("Results w/o ensemble: ", results)
    # logging.info(results)
    
    #print("probs_list = ", probs_list)
    #print("out_label_ids = ", out_label_ids)


    # print("shape(probs_list) = ", probs_list.shape)
    # print("shape(out_label_ids) = ", out_label_ids.shape)

    #predictii pentru clasa 1 (la mine 1 = same authors, 0 different)
    ua_scores_fn = "/data/darwin/bert/ua/scores_allseq_seq.pkl"
    ua_labels_fn = "/data/darwin/bert/ua/labels_allseq_seq.pkl"

    uf_scores_fn = "/data/darwin/bert/uf/scores_allseq_seq-fandom.pkl"
    uf_labels_fn = "/data/darwin/bert/uf/labels_allseq_seq-fandom.pkl"
    
    #load other scores
    ua_scores = np.array(pickle.load(open(ua_scores_fn, "rb")))
    ua_labels = pickle.load(open(ua_labels_fn, "rb"))
    ua_labels = np.stack(ua_labels).squeeze()
    print("ua_labels = ", ua_labels.shape)

    uf_scores = np.array(pickle.load(open(uf_scores_fn, "rb")))
    uf_labels = pickle.load(open(uf_labels_fn, "rb"))
    uf_labels = np.stack(uf_labels).squeeze()
    print("uf_labels = ", uf_labels.shape)

    # print("probs_list shape = ", probs_list.shape)
    # print("out_label_ids shape = ", out_label_ids.shape)
    # print("ua_scores shape = ", ua_scores.shape)
    
    #mean_scores = (ua_scores + probs_list) * 0.5
    #print("mean_scores = ", mean_scores)
    #results = evaluate_all(out_label_ids, mean_scores)
    results = evaluate_all(ua_labels, ua_scores)
    print("Results w/ ensemble: ", results)
    logging.info(results)
    # print("ua_scores = ", ua_scores)
    # print("ua_labels = ", ua_labels)

