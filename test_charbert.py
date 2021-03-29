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
    print("do_predict = ", args.do_predict)

    if not args.do_predict:
        print("do_predict is false")
        sys.exit(0)

    # config_args = json.load(open(args.config))
    # for k,v in config_args.items():
    #     setattr(args, k, v)

    # check output_dir
    if not os.path.exists(args.output_dir):
        print("output_dir does not exist")
        sys.exit(0)

    # set up logging
    args.output_dir = "/data/darwin/xs_closed_v1_2e-5_randchunk/checkpoint-11864"
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
    args.test_path = "/pan2020/closed_splits/closed_split_v1/xs/pan20-av-small-test"   
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
        do_train=False,
        do_eval=False,
        do_predict=True,
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
    
    # train model using Huggingface's Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=train_args,
    #     #data_collator=collate_fn,
    #     #train_dataset=train_dataset,
    #     #eval_dataset=val_dataset,
    #     #tokenizer=None,
    #     #model_init=None,
    #     compute_metrics=compute_pan_metrics
    #     #callbacks=[LogCallback]
    #     #optimizers=(optimizer, scheduler)
    # )
    #train_results = trainer.train()
    #logging.info("train results: %s", train_results)
    
    #val_results = trainer.evaluate()
    #print(val_results)

    # load best model on dev set
    #test_results = trainer.predict(test_dataset=test_dataset, metric_key_prefix='test')
    #print("test_results = ", test_results)
    
    args.eval_batch_size = 1
    results, preds_list = evaluate(args, test_dataset, model, test_mode=True)
    logging.info(results)
    print("Results = ", results)