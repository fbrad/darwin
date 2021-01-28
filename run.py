from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from torch.utils.data import DataLoader
from dataset_readers.pan2020_dataset import Pan2020Dataset
import torch

if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    #train_dataset = Pan2020Dataset('/pan2020/pan20-authorship-verification-training-small/train', tokenizer)
    #val_dataset = Pan2020Dataset('/pan2020/pan20-authorship-verification-training-small/val', tokenizer)
    dataset = Pan2020Dataset('/pan2020/pan20-authorship-verification-training-small/val', tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = ", device)

    def collate_fn(examples):
        return tokenizer.pad(examples, padding=True, return_tensors='pt')

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=3)
    #model = BertForSequenceClassification.from_pretrained('bert-base-cased')
    #model.train().to(device)

    for idx, batch in enumerate(dataloader):
        #print(batch)
        batch.to(device)
        print(batch['input_ids'].device)
        #outputs = model(**batch)
        #print(outputs)
        #print(outputs.loss.device)
        break