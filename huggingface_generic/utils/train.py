import numpy as np
import datetime
from tqdm import tqdm, trange
import time
import torch
from sklearn.metrics import classification_report, roc_auc_score

def calc_acc(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def evaluate_model(model, dataloader, device):
    total_loss = 0
    total_accuracy = 0

    all_preds = []
    all_scores = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Test'):

        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        b_input_mask = batch[2].to(device)

        with torch.no_grad():

            loss, logits = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

            total_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_accuracy += calc_acc(logits, label_ids)

            pred_flat = np.argmax(logits, axis=1).flatten()
            all_preds.append(pred_flat)
            all_scores.append(logits[:,1])
            all_labels.append(label_ids)

    avg_accuracy = total_accuracy / len(dataloader)
    avg_loss = total_loss / len(dataloader)

    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    clf_rep = classification_report(all_labels, all_preds, output_dict=True, target_names=['same', 'diff'])
    roc_auc = roc_auc_score(all_labels, all_scores)

    return roc_auc, clf_rep, avg_loss

def write_to_tb(name, roc_auc, clf_rep, tb_writer, step):
    tb_writer.add_scalar(f'{name}/roc_auc', roc_auc, step)

    tb_writer.add_scalar(f'{name}/avg_prec', clf_rep['weighted avg']['precision'], step)
    tb_writer.add_scalar(f'{name}/avg_rec', clf_rep['weighted avg']['recall'], step)
    tb_writer.add_scalar(f'{name}/avg_f1', clf_rep['weighted avg']['f1-score'], step)

def train_epoch(model, optimizer, train_dataloader, test_dataloader, device, tb_writer, global_step, eval_step=1000):
    total_train_loss = 0
    total_train_acc = 0
    batch_acc = 0

    all_preds = []
    all_scores = []
    all_labels = []

    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc='Train')):
        global_step += 1

        b_input_ids  = batch[0].to(device)
        b_labels     = batch[1].to(device)
        b_input_mask = batch[2].to(device)

        model.zero_grad()

        loss, logits = model(b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels
        )

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_train_acc += calc_acc(logits, label_ids)

        pred_flat = np.argmax(logits, axis=1).flatten()

        all_preds.append(pred_flat)
        all_scores.append(logits[:,1])
        all_labels.append(label_ids)

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if global_step % eval_step == 0:
            with torch.no_grad():
                model.eval()
                eval_roc, eval_clf_rep, avg_eval_loss = evaluate_model(model, test_dataloader, device)
                write_to_tb('test', eval_roc, eval_clf_rep, tb_writer, global_step)
                tb_writer.add_scalar(f'test/loss', avg_eval_loss, global_step)

    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    clf_rep = classification_report(all_labels, all_preds, output_dict=True, target_names=['same', 'diff'])
    roc_auc = roc_auc_score(all_labels, all_scores)

    write_to_tb('train', roc_auc, clf_rep, tb_writer, global_step)

    avg_train_loss = total_train_loss / len(train_dataloader)
    tb_writer.add_scalar(f'train/loss', avg_train_loss, global_step)

    return global_step