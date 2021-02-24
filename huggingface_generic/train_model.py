from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from datasets import PAN2020
from utils import train_epoch
import argparse
import torch
import time
import datetime
import numpy as np
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PAN20 Baseline')
parser.add_argument('--train_dir', type=str, default='/darkweb_ds/split_florin/xs/v2_split/pan20-av-small-notest.jsonl')
parser.add_argument('--test_dir', type=str, default='/darkweb_ds/split_florin/xs/v2_split/pan20-av-small-test.jsonl')
parser.add_argument('--tb_dir', type=str, default='./runs')
parser.add_argument('--model_type', type=str, default='bert-base-cased')
parser.add_argument('--exp_prefix', type=str, default='baseline')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=512)

args = parser.parse_args()
train_dir   = args.train_dir
test_dir    = args.test_dir
tb_dir      = args.tb_dir
model_type  = args.model_type
exp_prefix  = args.exp_prefix
lr          = args.lr
wd          = args.wd
batch_size  = args.batch_size
epochs      = args.epochs
seq_len     = args.seq_len

print('Done parsing args.')

tb_name = f'{tb_dir}/{exp_prefix}'
tb_writer = SummaryWriter(log_dir = tb_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModelForSequenceClassification.from_pretrained(model_type).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

train_ds = PAN2020(train_dir, tokenizer, seq_len, 3)
test_ds  = PAN2020(test_dir, tokenizer, seq_len, 3)

train_dataloader = DataLoader(
    train_ds,
    sampler = RandomSampler(train_ds),
    batch_size = batch_size
)

# train_dataloader = DataLoader(
#     test_ds,
#     sampler = RandomSampler(test_ds),
#     batch_size = batch_size
# )

test_dataloader = DataLoader(
    test_ds,
    sampler = RandomSampler(test_ds),
    batch_size = batch_size
)

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

model.train()

epochs_it = trange(epochs, desc='Epoch', mininterval=0)
global_step = 0

for e in epochs_it:
    global_step = train_epoch(model, optimizer, train_dataloader, test_dataloader, device, tb_writer, global_step=global_step)
