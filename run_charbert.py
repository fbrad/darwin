from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizerFast, BertTokenizer, TrainingArguments, Trainer, BertConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSequenceClassification
from transformers.models.bert import BertForSequenceClassification
from torch.utils.data import DataLoader
from models.character_bert import CharacterBertModel

from dataset_readers.pan2020_dataset import Pan2020Dataset
from utils.character_cnn import CharacterIndexer
import torch
import os
from tqdm import tqdm

if __name__ == '__main__':
    #DATA_TRAIN_PATH = "/pan2020/pan20-authorship-verification-training-small/train"
    #DATA_VAL_PATH = "/pan2020/pan20-authorship-verification-training-small/val"
    DATA_TRAIN_PATH = "data/train"
    DATA_VAL_PATH = "data/val"
    MODEL_TO_URL = {
        'general_character_bert': 'https://drive.google.com/open?id=11-kSfIwSWrPno6A4VuNFWuQVYD8Bg_aZ',
        'medical_character_bert': 'https://drive.google.com/open?id=1LEnQHAqP9GxDYa0I3UrZ9YV2QhHKOh2m',
        'general_bert': 'https://drive.google.com/open?id=1fwgKG2BziBZr7aQMK58zkbpI0OxWRsof',
        'medical_bert': 'https://drive.google.com/open?id=1GmnXJFntcEfrRY4pVZpJpg7FH62m47HS'
    }

    tokenizer = BertTokenizer.from_pretrained(
        os.path.join('pretrained_models', 'bert-base-uncased'),
        do_lower_case=False)
    tokenizer = tokenizer.basic_tokenizer
    characters_indexer = CharacterIndexer()
    tokenization_function = tokenizer.tokenize
    
    train_dataset = Pan2020Dataset(DATA_VAL_PATH, tokenizer=tokenizer, indexer=characters_indexer)
    val_dataset = Pan2020Dataset(DATA_VAL_PATH, tokenizer=tokenizer, indexer=characters_indexer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    config = BertConfig.from_pretrained(
        os.path.join('pretrained_models', "general_character_bert"),
        num_labels=2
    )
    model = BertForSequenceClassification(config=config)
    model.bert = CharacterBertModel.from_pretrained(
        os.path.join('pretrained_models', "general_character_bert"),
        config=config
    )
    model.train().to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    num_epochs = 1
    for idx in range(num_epochs):
        epoch_train_loss = 0.0
        for b_idx, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            batch = {k: v.to(device=device) for k, v in batch.items()}
            #print("batch = ", batch)
            outputs = model(**batch, return_dict=False)
            #print("outputs = ", outputs)
            loss = outputs['loss'] if 'loss' in outputs else outputs[0]
            print("Training loss ", loss)
            epoch_train_loss += loss
            loss.backward()
            optimizer.step()
            #break
        epoch_train_loss /= b_idx
            
        epoch_val_loss = 0.0
        for b_idx, batch in tqdm(enumerate(val_loader)):
            batch = {k: v.to(device=device) for k, v in batch.items()}
            outputs = model(**batch, return_dict=False)
            loss = outputs['loss'] if 'loss' in outputs else outputs[0]
            epoch_val_loss += loss
            break
        epoch_val_loss /= b_idx
        print("Epoch %d: train loss %f, val loss %f" % (idx, epoch_train_loss, epoch_val_loss))
