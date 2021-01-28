import json
import os
import numpy as np
from collections import defaultdict

def merge_data_and_labels(data_path: str, ground_truth_path: str, merged_data_path: str):
    """
    Merge authors data and ground truth data into a single file.
    """
    # read ground truth data
    with open(ground_truth_path) as f:
        labels = [json.loads(line) for line in f]
    
    out_fp = open(merged_data_path, "w")

    with open(data_path) as f:
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print(idx)
            entry = json.loads(line)
            assert entry["id"] == labels[idx]["id"], "id mismatch at idx = %d" % (idx)
            entry["same"] = labels[idx]["same"]
            entry["authors"] = labels[idx]["authors"]
            json.dump(entry, out_fp)
            out_fp.write('\n')

    out_fp.close()


def split_train_val(data_path: str, data_train_path: str, data_val_path: str, 
                    train_percentage: float, dataset_size: str = ['small', 'large']):
    """
    Split data intro train and val according to Araujo-Pino et al. 2020 
    ( https://pan.webis.de/downloads/publications/papers/araujopino_2020.pdf )
    Arguments:
        data_path: path to .jsonl file containing pairs of author texts
        data_train_path: path to .jsonl file where training pairs will be saved
        data_val_path: path to .jsonl file where validation pairs will be saved
        train_percentage: what percentage p in [0, 1] goes to training data
    """
    if os.path.exists(data_train_path):
        print("File ", data_train_path, " already exists")
        return
    
    if os.path.exists(data_val_path):
        print("File ", data_val_path, " already exists")
        return

    # small dataset: 52601  pairs, first 27834 entries are positive, first p% 
    # large dataset: 275565 pairs, first 147778 entries are positive
    # Dataset splitting order:
    # | positive val | positive train | negative val | negative train |
    # Say we want to split train/val as 90%-10%:
    # We go through the first 20% of positive examples:
    #  - half of them (10%) get assigned to train and half of them (10%) to val
    #  - the rest 80% get assigned to train   
    # This ensures that authors from positive pairs in val are equally present in train

    sizes = {
        "small": {"size": 52601, "positive": 27834, "negative": 24767},
        "large": {"size": 275565, "positive": 147778, "negative": 127787}
    }
    val_percentage = 1 - train_percentage
    assert val_percentage <= 0.5, "validation split too large"
    ds = sizes[dataset_size]
    with open(data_path) as f, open(data_train_path, "w") as train_fp, open(data_val_path, "w") as val_fp:
        # index where positive entries start to get assigned to train exclusively
        pos_train_start_idx = int(2 * val_percentage * ds['positive'])
        # index where negative entries start to get assigned to train
        neg_train_start_idx = ds['positive'] + int(val_percentage * ds['negative'])
 
        train_entries_count = 0
        val_entries_count = 0
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print("idx = ", idx)

            entry = json.loads(line)
            if entry['same']:
                if idx < pos_train_start_idx:
                    out_fp = val_fp if idx % 2 == 0 else train_fp
                else:
                    out_fp = train_fp
            else:
                out_fp = val_fp if idx < neg_train_start_idx else train_fp
            
            json.dump(entry, out_fp)
            out_fp.write('\n')

            # sanity check
            if out_fp == train_fp:
                train_entries_count += 1
            else:
                val_entries_count += 1
        
        print("train entries written: ", train_entries_count, ", val entries written: ", val_entries_count)

    return None

def split_large_jsonl(path_to_jsonl: str, output_folder):
    """
    Store each JSON line in a .jsonl file in a separate .json file in ```output_folder```.
    """
    if not os.path.exists(output_folder):
        print(output_folder, " folder does not exist")
        return
    
    if len(os.listdir(output_folder)) > 0:
        print(output_folder, " is not empty, abort")
        print(os.listdir(output_folder))
        return

    with open(path_to_jsonl) as fp:
        for idx, line in enumerate(fp):
            if idx % 1000 == 0:
                print(idx)
            entry = json.loads(line)
            entry_path = os.path.join(output_folder, entry["id"])
            if os.path.exists(entry_path):
                print(entry_path, " already exists")
            with open(entry_path, "w") as out_fp:
                json.dump(entry, out_fp)


if __name__ == '__main__':
    PAN2020_SMALL_DATA = "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
    PAN2020_SMALL_GT = "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"
    PAN2020_SMALL = "/pan2020/pan20-authorship-verification-training-small/pan20-av-small.jsonl"
    PAN2020_SMALL_TRAIN = "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-train.jsonl"
    PAN2020_SMALL_VAL = "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-val.jsonl"

    PAN2020_LARGE_DATA = "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large.jsonl"
    PAN2020_LARGE_GT = "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large-truth.jsonl"
    PAN2020_LARGE = "/pan2020/pan20-authorship-verification-training-large/pan20-av-large.jsonl"
    PAN2020_LARGE_TRAIN = "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-train.jsonl"
    PAN2020_LARGE_VAL = "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-val.jsonl"

    #merge_data_and_labels(PAN2020_DATA_LARGE, PAN2020_GT_LARGE, PAN2020_FULL_DATA_LARGE)
    #split_train_val(PAN2020_SMALL, PAN2020_SMALL_TRAIN, PAN2020_SMALL_VAL, 0.9, 'small')
    #split_train_val(PAN2020_LARGE, PAN2020_LARGE_TRAIN, PAN2020_LARGE_VAL, 0.95, 'large')
    split_large_jsonl(PAN2020_SMALL_VAL, "/pan2020/pan20-authorship-verification-training-small/val")
    # sanity checks