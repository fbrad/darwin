from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizerFast, BertTokenizer, TrainingArguments, Trainer, BertConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSequenceClassification
from transformers.models.bert import BertForSequenceClassification
from transformers import BertModel
from transformers.optimization import get_constant_schedule
from torch.utils.data import DataLoader
from torch.optim import SGD
from models.character_bert import CharacterBertModel
from models.character_bert_for_classification import CharacterBertForSequenceClassification
from dataset_readers.pan2020_dataset import Pan2020Dataset
from dataset_readers.pan2020_segments_dataset import Pan2020SegmentsDataset
from utils.training import train, PanTrainer, LogCallback
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


if __name__ == '__main__':   
    # read config file and merge with arguments
    args = parse_args()
    config_args = json.load(open(args.config))
    for k,v in config_args.items():
        setattr(args, k, v)

    # Set up logging
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'output.log'),
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
    if args.backbone == 'charbert':
        tokenizer = tokenizer.basic_tokenizer
        characters_indexer = CharacterIndexer()       
        #train_dataset = Pan2020SegmentsDataset(
        train_dataset = Pan2020Dataset(
            args.train_path, 
            tokenizer=tokenizer, 
            indexer=characters_indexer, 
            debug=args.debug
        )
        #val_dataset = Pan2020SegmentsDataset(
        val_dataset = Pan2020Dataset(
            args.val_path, 
            tokenizer=tokenizer, 
            indexer=characters_indexer,
            debug=args.debug
        )
    else:
        train_dataset = Pan2020Dataset(
            args.train_path, 
            tokenizer=tokenizer,
            debug=args.debug
        )
        val_dataset = Pan2020Dataset(
            args.val_path, 
            tokenizer=tokenizer,
            debug=args.debug
        )

    num_train_steps_per_epoch = math.ceil(len(train_dataset) / args.train_batch_size)
    num_train_steps = num_train_steps_per_epoch * args.num_train_epochs
    num_warmup_steps = int(args.warmup_ratio * num_train_steps)

    # set up backbone
    if args.backbone == 'charbert':
        logging.info('Loading %s model', "general_character_bert")
        config = BertConfig.from_pretrained(
            os.path.join('pretrained_models', "general_character_bert"),
            num_labels=2
        )
        config.update({"return_dict": False})
        model = BertForSequenceClassification(config=config)
        #model = CharacterBertForSequenceClassification(config=config)
        model.bert = CharacterBertModel.from_pretrained(
            os.path.join('pretrained_models', "general_character_bert"),
            config=config
        )
    else:
        logging.info('Loading %s model', "general_bert")
        config = BertConfig.from_pretrained(
            os.path.join('pretrained_models', "bert-base-uncased"),
            num_labels=2
        )
        config.update({"return_dict": False})
        model = BertForSequenceClassification(config=config)
        model.bert = BertModel.from_pretrained(
            os.path.join('pretrained_models', "bert-base-uncased"),
            config=config
        )
    model.to(args.device)

    if args.freeze_backbone:
        # freeze all params except classification weights
        for param_name, param in model.named_parameters():
            if not 'classifier' in param_name:
                param.requires_grad = False

        for param_name, param in model.named_parameters():
            logging.info("Param %s, requires_grad = %s, size: %s" % (param_name, param.requires_grad, param.size()))

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
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        do_predict=False,
        disable_tqdm=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        #lr_scheduler_type="constant",
        lr_scheduler_type="linear",
        warmup_steps=num_warmup_steps,
        logging_dir=args.output_dir,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='overall',
        greater_is_better=True
    )
    
    # train model using Huggingface's Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        #data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #tokenizer=None,
        #model_init=None,
        compute_metrics=compute_pan_metrics
        #callbacks=[LogCallback]
        #optimizers=(optimizer, scheduler)
    )
    train_results = trainer.train()
    print(train_results)
    logging.info("train results: %s", train_results)
    
    #val_results = trainer.evaluate()
    #print(val_results)

