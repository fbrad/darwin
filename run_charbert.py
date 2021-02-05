from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizerFast, BertTokenizer, TrainingArguments, Trainer, BertConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSequenceClassification
from transformers.models.bert import BertForSequenceClassification
from torch.utils.data import DataLoader
from models.character_bert import CharacterBertModel
from dataset_readers.pan2020_dataset import Pan2020Dataset
from utils.training import train
from utils.character_cnn import CharacterIndexer
from utils.misc import set_seed, parse_args
import sys
import math
import torch
import logging
import os
from tqdm import tqdm

# def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#     print("[on_step_begin] kwargs = ", kwargs)

if __name__ == '__main__':
    #DATA_TRAIN_PATH = "/pan2020/pan20-authorship-verification-training-small/train"
    #DATA_VAL_PATH = "/pan2020/pan20-authorship-verification-training-small/val"
    DATA_TRAIN_PATH = "data/train"
    DATA_VAL_PATH = "data/val"
    args = parse_args()
    args.debug = True
    args.output_dir = "output/character_bert"

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'output.log'),
        format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        filemode='w',
        force=True
    )
    logging.getLogger().setLevel(logging.INFO)

    # set up tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join('pretrained_models', 'bert-base-uncased'),
        do_lower_case=args.do_lower_case)
    tokenizer = tokenizer.basic_tokenizer
    characters_indexer = CharacterIndexer()
    tokenization_function = tokenizer.tokenize
    
    train_dataset = Pan2020Dataset(
        DATA_TRAIN_PATH, 
        tokenizer=tokenizer, 
        indexer=characters_indexer, 
        debug=args.debug
    )
    val_dataset = Pan2020Dataset(
        DATA_VAL_PATH, 
        tokenizer=tokenizer, 
        indexer=characters_indexer,
        debug=args.debug
    )
    num_train_steps_per_epoch = math.ceil(len(train_dataset) / args.train_batch_size)
    num_train_steps = num_train_steps_per_epoch * args.num_train_epochs
    num_warmup_steps = int(args.warmup_ratio * num_train_steps)

    # set up model
    logging.info('Loading %s model', "general_character_bert")
    config = BertConfig.from_pretrained(
        os.path.join('pretrained_models', "general_character_bert"),
        num_labels=2
    )
    model = BertForSequenceClassification(config=config)
    model.bert = CharacterBertModel.from_pretrained(
        os.path.join('pretrained_models', "general_character_bert"),
        config=config
    )
    model.to(args.device)



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
        do_train=False,
        do_eval=False,
        do_predict=False,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        #eval_accumulation_steps=int,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="linear",
        warmup_steps=num_warmup_steps,
        logging_dir=args.output_dir,
        label_names="labels",
        #load_best_model_at_end=False,
        #metric_for_best_model='overall',
        #greater_is_better=True,
    )

    # TODO: Use Trainer from huggingface
    trainer = Trainer(
        model=model,
        args=train_args,
        #data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #tokenizer=None,
        #model_init=None,
        #compute_metrics=eval_pan,
        callbacks=None,
        #optimizers=None
    )
    train_results = trainer.train()
    print(train_results)
    val_results = trainer.evaluate()
    print(val_results)

