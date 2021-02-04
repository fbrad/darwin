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
import torch
import logging
import os
from tqdm import tqdm

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

    global_step, train_loss, best_val_metric, best_val_epoch = train(
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        model=model
    )
    logging.info("global_step = %s, average training loss = %s", global_step, train_loss)
    logging.info("Best performance: Epoch=%d, Value=%s", best_val_epoch, best_val_metric)

    # TODO: move this into trainer
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # num_epochs = 1
    # for idx in range(num_epochs):
    #     epoch_train_loss = 0.0
    #     for b_idx, batch in tqdm(enumerate(train_loader)):
    #         optimizer.zero_grad()
    #         batch = {k: v.to(device=device) for k, v in batch.items()}
    #         outputs = model(**batch, return_dict=False)
    #         loss = outputs['loss'] if 'loss' in outputs else outputs[0]
    #         #print("Training loss ", loss)
    #         epoch_train_loss += loss
    #         loss.backward()
    #         optimizer.step()
    #         #break
    #     epoch_train_loss /= b_idx
            
    #     epoch_val_loss = 0.0
    #     for b_idx, batch in tqdm(enumerate(val_loader)):
    #         batch = {k: v.to(device=device) for k, v in batch.items()}
    #         outputs = model(**batch, return_dict=False)
    #         loss = outputs['loss'] if 'loss' in outputs else outputs[0]
    #         epoch_val_loss += loss
    #         break
    #     epoch_val_loss /= b_idx
    #     print("Epoch %d: train loss %f, val loss %f" % (idx, epoch_train_loss, epoch_val_loss))
