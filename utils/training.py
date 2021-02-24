# Functions are adapted from Huggingface's transformers library:
# https://github.com/huggingface/transformers

""" Defines training and evaluation functions. """
import os
import logging
import datetime

import tqdm
import numpy as np
import sklearn.metrics as sklearn_metrics

import torch
from torch.nn.functional import softmax
from torch.utils.data import SequentialSampler, RandomSampler, DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollator
from utils.metrics import evaluate_all
from utils.misc import set_seed
import logging
from typing import Union, Optional, Callable, Tuple, Dict, List


class LogCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_loss = 0.0
        self._epoch_steps = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Unfortunately, this events gets called after TensorboardCallBack.on_log(), so the 
        "epoch_loss" never gets logged.
        """
        if not control.should_evaluate and 'loss' in kwargs['logs']:
            logs = kwargs['logs']
            self._epoch_loss += logs['loss']
            self._epoch_steps += 1
            if logs['epoch'].is_integer():
                logs['loss_epoch'] = self._epoch_loss / self._epoch_steps
                self._epoch_loss = 0.0
                self._epoch_steps = 0

class PanTrainer(Trainer):
    def __init__(self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(model=model, args=args, data_collator=data_collator, 
            train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, 
            model_init=model_init, compute_metrics=compute_metrics, callbacks=callbacks,
            optimizers=optimizers
        )

    # def evaluate(
    #     self,
    #     eval_dataset: Optional[Dataset] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> Dict[str, float]:
    #     metrics = super().evaluate(eval_dataset=eval_dataset, 
    #         ignore_keys=ignore_keys, 
    #         metric_key_prefix=metric_key_prefix
    #     )
    #     #print("[PanTrainer:evalute] metrics = ", metrics)
    #     return metrics

    # def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
    #     print("haha")
    #     super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch)
    #     if self.control.should_log:
    #         print("[PanTrainer._maybe_log...] self.state = ", self.state)

def train(args, train_dataset, eval_dataset, model):
    """ Trains the given model on the given dataset. """

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size
    )

    n_train_steps__single_epoch = len(train_dataloader)
    n_train_steps = n_train_steps__single_epoch * args.num_train_epochs
    args.logging_steps = n_train_steps__single_epoch

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio*n_train_steps),
        num_training_steps=n_train_steps
    )

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Total optimization steps = %d", n_train_steps)
    logging.info("  Using linear warmup (ratio=%s)", args.warmup_ratio)
    logging.info("  Using weight decay (value=%s)", args.weight_decay)
    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    best_metric, best_epoch = -1.0, -1  # Init best -1 so that 0 > best

    model.zero_grad()
    train_iterator = tqdm.trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    set_seed(seed_value=args.seed)  # Added here for reproductibility
    for num_epoch in train_iterator:
        epoch_loss = 0.0
        epoch_iterator = tqdm.tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = {k: v.to(device=args.device) for k, v in batch.items()}
            batch['return_dict'] = False

            outputs = model(**batch)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss.backward()
            tr_loss += loss.item()
            epoch_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0:
                # Log metrics
                # -- Only evaluate when single GPU otherwise metrics may not average well
                results, _ = evaluate(
                    args=args,
                    eval_dataset=eval_dataset,
                    model=model
                )

                logging_loss = tr_loss
                metric = results['overall']

                if metric > best_metric:
                    best_metric = metric
                    best_epoch = num_epoch

                    # Save model checkpoint
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    model.save_pretrained(args.output_dir)
                    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                    logging.info("Saving model checkpoint to %s", args.output_dir)

                    #torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
                    #torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
                    #logging.info("Saving optimizer and scheduler states to %s", args.output_dir)
        logging.info(" epoch loss %d = %f", num_epoch, epoch_loss / step + 1)

    return global_step, tr_loss / global_step, best_metric, best_epoch


def evaluate(args, eval_dataset, model):
    """ Evaluates the given model on the given dataset. """

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size)

    # Evaluate!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k: v.to(device=args.device) for k, v in batch.items()}
        batch['return_dict'] = False

        with torch.no_grad():
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        if preds is None:
            # batch x 2
            probs = softmax(logits, dim=1).detach().cpu().numpy()
            preds = logits.detach().cpu().numpy()
            # batch
            out_label_ids = batch["labels"].detach().cpu().numpy()
        else:
            # num_eval_samples x 2
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            probs = np.append(probs, softmax(logits, dim=1).detach().cpu().numpy(), axis=0)
            # num_eval_samples
            out_label_ids = np.append(out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    
    # num_eval_samples
    preds_list = np.argmax(preds, axis=1)
    probs_list = probs[:,1]

    results = evaluate_all(out_label_ids, probs_list)
    results['loss'] = eval_loss

    # results = {
    #     "loss": eval_loss,
    #     "precision": sklearn_metrics.precision_score(out_label_ids, preds_list, average='micro'),
    #     "recall": sklearn_metrics.recall_score(out_label_ids, preds_list, average='micro'),
    #     "f1": sklearn_metrics.f1_score(out_label_ids, preds_list, average='micro'),
    #     "accuracy": sklearn_metrics.accuracy_score(out_label_ids, preds_list),
    # }

    logging.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logging.info("  %s = %s", key, str(results[key]))

    return results, preds_list
