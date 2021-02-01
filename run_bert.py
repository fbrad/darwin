from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizerFast, TrainingArguments, Trainer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from dataset_readers.pan2020_dataset import Pan2020Dataset
import torch
from tqdm import tqdm

if __name__ == '__main__':
    DATA_TRAIN_PATH = "/pan2020/pan20-authorship-verification-training-small/train"
    DATA_VAL_PATH = "/pan2020/pan20-authorship-verification-training-small/val"
    #DATA_TRAIN_PATH = "data/train"
    #DATA_VAL_PATH = "data/val"

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset = Pan2020Dataset(DATA_VAL_PATH, tokenizer)
    val_dataset = Pan2020Dataset(DATA_VAL_PATH, tokenizer)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = ", device)

    def collate_fn(examples):
        return tokenizer.pad(examples, padding=True, return_tensors='pt')

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=8)
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.train().to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    num_epochs = 20
    for idx in range(num_epochs):
        epoch_train_loss = 0.0
        for b_idx, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            batch.to(device)
            outputs = model(**batch)
            loss = outputs['loss']
            print("Training loss ", loss)
            epoch_train_loss += loss
            loss.backward()
            optimizer.step()
        epoch_train_loss /= b_idx
            
        epoch_val_loss = 0.0
        for b_idx, batch in tqdm(enumerate(val_loader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs['loss']
            epoch_val_loss += loss
        epoch_val_loss /= b_idx
        print("Epoch %d: train loss %f, val loss %f" % (idx, epoch_train_loss, epoch_val_loss))
