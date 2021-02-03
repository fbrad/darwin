""" Miscellaneous utils. """
import random
import logging
import torch
import numpy as np
import argparse
import datetime

MODEL_TO_URL = {
    'general_character_bert': 'https://drive.google.com/open?id=11-kSfIwSWrPno6A4VuNFWuQVYD8Bg_aZ',
    'medical_character_bert': 'https://drive.google.com/open?id=1LEnQHAqP9GxDYa0I3UrZ9YV2QhHKOh2m',
    'general_bert': 'https://drive.google.com/open?id=1fwgKG2BziBZr7aQMK58zkbpI0OxWRsof',
    'medical_bert': 'https://drive.google.com/open?id=1GmnXJFntcEfrRY4pVZpJpg7FH62m47HS'
}

def set_seed(seed_value):
    """ Sets the random seed to a given value. """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logging.info("Random seed: %d", seed_value)

def parse_args():
    AVAILABLE_MODELS = ['general_character_bert', 'medical_character_bert', 'general_bert', 'medical_bert']
    """ Parse command line arguments and initialize experiment. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding",
        type=str,
        default='general_character_bert',
        choices=AVAILABLE_MODELS,
        help="The model to use."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to apply lowercasing during tokenization."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size to use for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2,
        help="Batch size to use for evaluation."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_ratio",
        default=0.1, type=int, help="Linear warmup over warmup_ratio*total_steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Do training & validation."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Do prediction on the test set."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()
    args.start_time = datetime.datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')

    # --------------------------------- INIT ---------------------------------

    # Set up logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
    #     datefmt="%d/%m/%Y %H:%M:%S",
    #     level=logging.INFO)

    # Check for GPUs
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1  # This script doesn't support multi-gpu
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # Set random seed for reproducibility
    set_seed(seed_value=args.seed)

    return args
