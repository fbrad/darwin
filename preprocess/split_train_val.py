import json
import os
import random
import numpy as np
from collections import defaultdict
from typing import Dict, Union, List, Callable, Tuple

def merge_data_and_labels(data_path: str, ground_truth_path: str, merged_data_path: str):
    """
    Merge PAN authors data and ground truth data into a single ```.jsonl``` file.
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


def remove_text_from_jsonl(input_jsonl: str, output_jsonl: str):
    """
    Remove text from .jsonl to get smaller files.
    Args:
        input_jsonl (str): path to original .jsonl file
        output_jsonl (str): path to 
    """
    with open(input_jsonl) as in_fp, open(output_jsonl, "w") as out_fp:
        for idx, line in enumerate(in_fp):
            if idx % 1000 == 0:
                print("idx = ", idx)
            entry = json.loads(line)
            del entry['pair']
            json.dump(entry, out_fp)
            out_fp.write('\n')

def write_jsonl_to_folder(path_to_jsonl: str, output_folder):
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

    duplicate_count = 0
    with open(path_to_jsonl) as fp:
        for idx, line in enumerate(fp):
            if idx % 1000 == 0:
                print(idx)
            entry = json.loads(line)
            entry_path = os.path.join(output_folder, entry["id"] + ".json")
            if os.path.exists(entry_path):
                duplicate_count += 1

            with open(entry_path, "w") as out_fp:
                json.dump(entry, out_fp)
                out_fp.write("\n")

    print("Duplicates: ", duplicate_count)


def read_jsonl_examples(data_path: str) -> List[Dict]:
    """
    Reads example pairs from a ```.jsonl``` file and returns them as a list of dictionaries.

    Args:
        data_path (str): path to .jsonl file containing the documents pairs, one per line
    Returns:
        List[Dict]: the dataset
    """
    examples = []
    with open(data_path) as fp:
        for idx, line in enumerate(fp):
            if idx % 10000 == 0:
                print("[read_jsonl_examples] read %d examples" % (idx))
            examples.append(json.loads(line))

    return examples

def get_authors_data_from_folder(authors_folder: str, num_authors: int=-1) -> Dict:
    """
    Read authors from folder into a dictionary:
        {"$author_id": [{"fandom" : ..., "text": ....}]}
    """
    authors_data = defaultdict(list)
    author_files = os.listdir(authors_folder)
    if num_authors > 0:
        author_files = author_files[:num_authors]

    for idx, author_file in enumerate(author_files):
        if idx % 10000 == 0:
            print("[get_authors_data_from_folder] processed ", idx, " authors")
        author_id = author_file[:-6]
        with open(os.path.join(authors_folder, author_file)) as fp:
            author_data = []
            for line in fp:
                author_data.append(json.loads(line))
        authors_data[author_id] = author_data

    return authors_data

def get_authors_data_from_jsonl(path_to_jsonl: str, pan_authors_folder: str):
    """
    Get authors and their documents from the .jsonl doc pairs.
    For each author, save its documents in ```$author_id.jsonl``` inside
    the ```pan_authors_folder```. 
    Args:
        path_to_jsonl (str): [description]
        pan_authors_folder (str): [description]
    """
    authors_to_docs = defaultdict(list)
    with open(path_to_jsonl) as fp:
        for idx, line in enumerate(fp):
            if idx % 1000 == 0:
                print(idx)
            entry = json.loads(line)

            a1, a2 = entry['authors'][0], entry['authors'][1]
            d1, d2 = entry['pair'][0], entry['pair'][1]
            f1, f2 = entry['fandoms'][0], entry['fandoms'][1]
            for (a,f,d) in zip([a1, a2], [f1,f2], [d1, d2]):
                duplicate = False
                for entry in authors_to_docs[a]:
                    if entry['text'] == d:
                        duplicate = True
                        break
                if not duplicate:
                    authors_to_docs[a].append({'fandom': f, 'text': d})

    # write author data to ```pan_authors_folder```
    for author, docs in authors_to_docs.items():
        author_path = os.path.join(pan_authors_folder, author + ".jsonl")
        with open(author_path, "w") as fp:
            for doc in docs:
                json.dump(doc, fp)
                fp.write('\n')


def split_pan_dataset_closed_set_v1(examples: List[Dict], 
                                    test_split_percentage: float) -> (List, List):
    """
    Split data intro train and val/test in an almost closed-set fashion, by 
    following Araujo-Pino et al. 2020 
    ( https://pan.webis.de/downloads/publications/papers/araujopino_2020.pdf )
    This algorithm ensures that authors of SA pairs in the test set appear in the
    training set, but gives no guarantee that authors of test DA pairs appear in 
    the training set.
    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    """
    assert test_split_percentage > 0 and test_split_percentage < 1, "test size in (0,1)"
    sizes = {
        "small": {"size": 52601, "positive": 27834, "negative": 24767},
        "large": {"size": 275565, "positive": 147778, "negative": 127787}
    }

    test_ids = []
    for idx, example in enumerate(examples):
        if idx % 10000 == 0:
            print("[split_pan_dataset_closed_set_v1] processed %d examples" % (idx))
        
    # determine Train/Test sizes
    same_author_examples = [ex for ex in examples if ex['same']]
    diff_author_examples = [ex for ex in examples if not ex['same']]
    random.shuffle(diff_author_examples)
    sa_size = len(same_author_examples)
    da_size = len(diff_author_examples)
    sa_test_size = int(test_split_percentage * sa_size)
    sa_train_size = sa_size - sa_test_size
    da_test_size = int(test_split_percentage * da_size)
    da_train_size = da_size - da_test_size

    # retrieve documents of all same-author (SA) pairs
    # sa_docs = {'author_id': [ids of SA pairs of this authors]}
    sa_docs = defaultdict(list)
    for example in same_author_examples:
        author_id = example['authors'][0] 
        sa_docs[author_id].append(example['id'])

    # first, populate SA test set
    print("[split_pan_dataset_closed_set_v1] Adding same-author (SA) pairs to the test set")
    sa_test_count = 0
    for author_id, pair_ids in sa_docs.items():
        author_docs_num = len(pair_ids)
        if author_docs_num >= 2:
            test_ids += pair_ids[:author_docs_num // 2]
            sa_test_count += author_docs_num // 2
        if sa_test_count >= sa_test_size:
            break
    
    # add DA pairs to test set
    da_test_count = 0
    print("[split_pan_dataset_closed_set_v1] Adding different-author (DA) pairs to the test set")
    for idx, example in enumerate(diff_author_examples):
        if idx % 10000 == 0:
            print("[split_pan_dataset_closed_set_v1] processed %d examples" % (idx))
        test_ids.append(example['id'])
        da_test_count += 1
        if da_test_count == da_test_size:
            break

    test_ids_map = {test_id:1 for test_id in test_ids}
    train_ids = [example['id'] for example in examples if example['id'] not in test_ids_map]
    train_ids_map = {train_id:1 for train_id in train_ids}

    # statistics
    train_stats = defaultdict(int)
    train_stats['size'] = len(train_ids)
    test_stats = defaultdict(int)
    test_stats['size'] = len(test_ids)
    for example in examples:
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        stats_dict = train_stats if example['id'] in train_ids_map else test_stats
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
    
    for split_name, stats_dict in zip(["TRAIN", "TEST"], [train_stats, test_stats]):
        print("%s size: %d" % (split_name, stats_dict['size']))
        print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
        print("        Same fandom pairs: ", stats_dict['sa_sf'])
        print("        Different fandom pairs: ", stats_dict['sa_df'])
        print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
        print("        Same fandom pairs: ", stats_dict['da_sf'])
        print("        Different fandom pairs: ", stats_dict['da_df'])

    return (train_ids, test_ids)


def split_pan_dataset_closed_set_v2(examples: List[Dict], 
                                    test_split_percentage: float) -> (List, List):
    """
    Split PAN 2020 dataset in Train and Test under the closed-set assumption*. This requires that
    authors in Test set appear in Train as well. However, due to the large number of authors 
    in the different-author (DA) pairs, it is difficult to achieve this strictly. We try to 
    make sure that at least one of the authors (ai, aj) in DA Test pairs appears in DA Train 
    pairs or in same-author (SA) Train pairs.

    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    
    Returns a list of unique pair ids for each dataset split
    """
    assert test_split_percentage > 0 and test_split_percentage < 1, "test size in (0,1)"
    sizes = {
        "small": {"size": 52601, "positive": 27834, "negative": 24767},
        "large": {"size": 275565, "positive": 147778, "negative": 127787}
    }

    # determine Train/Test sizes
    same_author_examples = [ex for ex in examples if ex['same']]
    diff_author_examples = [ex for ex in examples if not ex['same']]
    random.shuffle(diff_author_examples)
    sa_size = len(same_author_examples)
    da_size = len(diff_author_examples)
    sa_test_size = int(test_split_percentage * sa_size)
    sa_train_size = sa_size - sa_test_size
    da_test_size = int(test_split_percentage * da_size)
    da_train_size = da_size - da_test_size
    
    # add test ids of same-author pairs
    test_ids = []
   
    # retrieve author ids of same-author (SA) pairs
    sa_authors_ids = set()
    for example in same_author_examples:
        sa_authors_ids.add(example['authors'][0])
    
    # Algorithm
    # Create dictionary of frequencies of each author in DA pairs
    # freq = {'a1': 5, 'a2': 4, 'a3': 1, ..., etc}
    # Go through the DA pairs (ai, aj)
    #   if (ai, aj) appear in other DA pairs
    #       move (ai, aj) to test and
    #       decrease frequencies
    #   else if (ai, aj) appear in SA test pairs:
    #       move (ai, aj) to test
    #   else
    #       move (ai, aj) to train
    
    # create frequency of author ids in DA pairs    
    diff_author_freq = defaultdict(int)
    for example in diff_author_examples:
        fst_author_id = example['authors'][0]
        snd_author_id = example['authors'][1]
        diff_author_freq[fst_author_id] += 1
        diff_author_freq[snd_author_id] += 1

    # populate test set with DA pairs (ai, aj) such that at least one of the authors 
    # ai or aj in the test split appears in other DA train pairs or in SA pairs
    test_ids = []
    test_author_ids = set()
    da_sf = 0
    for example in diff_author_examples:
        fst_author_id = example['authors'][0] # a1
        snd_author_id = example['authors'][1] # a2
        same_fandom = example['fandoms'][0] == example['fandoms'][1]

        # check if a1 or a2 appear in other DA pairs
        fst_frequent = diff_author_freq[fst_author_id] >= 2
        snd_frequent = diff_author_freq[snd_author_id] >= 2
        if fst_frequent or snd_frequent:
            test_ids.append(example['id'])
            if same_fandom:
                da_sf += 1
            if fst_frequent:
                test_author_ids.add(fst_author_id)
                diff_author_freq[fst_author_id] -= 1
            if snd_frequent:
                test_author_ids.add(snd_author_id)
                diff_author_freq[snd_author_id] -= 1
        # check if a1 or a2 appear in SA pairs
        elif fst_author_id in sa_authors_ids or snd_author_id in sa_authors_ids:
            test_ids.append(example['id'])
            if same_fandom:
                da_sf += 1
            if fst_author_id in sa_authors_ids:
                test_author_ids.add(fst_author_id)
            if snd_author_id in sa_authors_ids:
                test_author_ids.add(snd_author_id)

    da_ids = [example['id'] for example in diff_author_examples]
    train_ids = [ex_id for ex_id in da_ids if ex_id not in test_ids]

    print("Number of different-author (DA) pairs: ", len(diff_author_examples))
    print("    Number of candidate DA test pairs: ", len(test_ids))
    print("         of which same fandom: ", da_sf)
    print("         of which diff fandom: ", len(test_ids) - da_sf)
    print("    Number of candidate DA train pairs: ", len(train_ids))

    # if too many DA test candidates, trim them
    if len(test_ids) > da_test_size:
        print("We only need %d DA test examples, trimming %d examples" % \
            (da_test_size, len(test_ids)-da_test_size))
        test_ids = test_ids[:da_test_size]
        # update author ids in test
        test_author_ids = set()
        for example in diff_author_examples:
            if example['id'] in test_ids:
                test_author_ids.add(example['authors'][0])
                test_author_ids.add(example['authors'][1])
    else:
        # if not enough DA test pairs, add further DA pairs
        da_test_count = len(test_ids)
        print("Not enough DA test examples %d/%d, adding other pairs" % \
            (da_test_count, da_test_size))
        for example in diff_author_examples:
            if example['id'] not in test_ids:
                test_ids.append(example['id'])
                test_author_ids.add(example['authors'][0])
                test_author_ids.add(example['authors'][1])
                da_test_count += 1
                if da_test_count == da_test_size:
                    break

    # retrieve documents of all same-author (SA) pairs
    # sa_docs = {'author_id': [ids of SA pairs of this authors]}
    sa_docs = defaultdict(list)
    for example in same_author_examples:
        author_id = example['authors'][0] 
        sa_docs[author_id].append(example['id'])

    # first, populate SA test set with authors belonging to DA test set
    print("Adding same-author (SA) pairs to the test set (authors from DA)")
    sa_test_count = 0
    for author_id, pair_ids in sa_docs.items():
        if author_id not in test_author_ids:
            continue

        author_docs_num = len(pair_ids)
        if author_docs_num >= 2:
            test_ids += pair_ids[:author_docs_num // 2]
            test_author_ids.add(author_id)
            sa_test_count += author_docs_num // 2
        if sa_test_count >= sa_test_size:
            break
    
    if sa_test_count >= sa_test_size:
        print("Added SA examples to test set: %d/%d" % (sa_test_count, sa_test_size))
    else:
        print("Not enough SA examples in test set: %d/%d, adding others" % (sa_test_count, sa_test_size))
        # if not enough, populate SA test set with other authors as well

        for author_id, pair_ids in sa_docs.items():
            # we have already added this author
            if author_id in test_author_ids:
                continue

            author_docs_num = len(pair_ids)
            if author_docs_num >= 2:
                test_ids += pair_ids[:author_docs_num // 2]
                test_author_ids.add(author_id)
                sa_test_count += author_docs_num // 2
            if sa_test_count >= sa_test_size:
                break
        print("Completed SA examples in test set: %d/%d" % (sa_test_count, sa_test_size))

    
    test_ids_map = {test_id:1 for test_id in test_ids}
    train_ids = [example['id'] for example in examples if example['id'] not in test_ids_map]
    train_ids_map = {train_id:1 for train_id in train_ids}

    # statistics
    train_stats = defaultdict(int)
    test_stats = defaultdict(int)
    for example in examples:
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        stats_dict = train_stats if example['id'] in train_ids_map else test_stats
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
    
    for split_name, stats_dict in zip(["TRAIN", "TEST"], [train_stats, test_stats]):
        split_ids = train_ids if split_name == 'TRAIN' else test_ids
        print("%s size: %d" % (split_name, len(split_ids)))
        print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
        print("        Same fandom pairs: ", stats_dict['sa_sf'])
        print("        Different fandom pairs: ", stats_dict['sa_df'])
        print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
        print("        Same fandom pairs: ", stats_dict['da_sf'])
        print("        Different fandom pairs: ", stats_dict['da_df'])

    return (train_ids, test_ids)


def split_pan_dataset_open_set_unseen_fandoms(examples: List[Dict], 
                                              test_split_percentage:float) -> (List, List):
    """
    Split dataset into train in test such that fandoms from train
    do not appear in test.
    Algorithm:
        1. Let F be the fandoms of same-author (SA) pairs
        2. Split F into two disjoint sets F_train and F_test
        2. populate test set with SA pairs from F_test until enough examples
        5. iterate through DA pairs (a1, a2, f1, f2)
           add them to test set if f1 and f2 not in F_train
    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    
    Returns a list of unique pair ids for each dataset split
    """

    sa_examples = [ex for ex in examples if ex['same']]
    random.shuffle(sa_examples)
    sa_size = len(sa_examples)
    sa_test_size = int(test_split_percentage * sa_size)
    sa_train_size = sa_size - sa_test_size

    da_examples = [ex for ex in examples if not ex['same']]
    da_sf_examples = [ex for ex in da_examples if ex['fandoms'][0] == ex['fandoms'][1]]
    da_df_examples = [ex for ex in da_examples if ex['fandoms'][0] != ex['fandoms'][1]]
    da_size = len(da_examples)
    da_sf_size = len(da_sf_examples)
    da_df_size = da_size - da_sf_size

    da_sf_test_size = int(test_split_percentage * da_sf_size)
    da_sf_train_size = da_sf_size - da_sf_test_size
    da_df_test_size = int(test_split_percentage * da_df_size)
    da_df_train_size = da_df_size - da_df_test_size

    da_test_size = da_sf_test_size + da_df_test_size
    da_train_size = da_sf_train_size + da_df_train_size

    test_ids = []
    # authors_g1 = {"author_id": {"ids": [$id, $id, ,,,],
    #                             "fandoms: [f1, f2, ...]}
    #              }
    # fandoms_train = {"fandom": {"ids": [$id, $id, ,,,],
    #                             "authors: [a1, a2, ...]}
    #              }
    #authors_g1 = defaultdict(dict)
    fandoms_da_sf_train, fandoms_da_sf_test = defaultdict(dict), {}
    authors_da_sf_train = {}
    for idx, example in enumerate(da_sf_examples):
        if idx % 10000 == 0:
            print("[open_set_unseen_fandoms] processed %d DA-SF examples" % (idx))
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        
        for a,f in zip([a1, a2], [f1, f1]):
            if f not in fandoms_da_sf_train:
                fandoms_da_sf_train[f]['ids'] = [example['id']]
                fandoms_da_sf_train[f]['authors'] = set([a])
            else:
                fandoms_da_sf_train[f]['ids'].append(example['id'])
                fandoms_da_sf_train[f]['authors'].add(a)

    print("[open_set_unseen_fandoms] #fandoms in DA-SF = ", len(fandoms_da_sf_train))
    # populate test set with DA-SF examples
    da_sf_test_count = 0
    authors_da_sf_test = {}
    least_freq_fandoms_train = sorted(fandoms_da_sf_train.items(), key=lambda x: len(x[1]['ids']))
    # for fandom, fandom_info in least_freq_fandoms_train:
    #     print("Fandom %s size %d" % (fandom, len(fandom_info['ids'])))
    for fandom, fandom_info in least_freq_fandoms_train:
        for a in fandom_info['authors']:
            authors_da_sf_test[a] = 1
        for pair_id in fandom_info['ids']:
            test_ids.append(pair_id)
        
        # move fandom info to the fandom test group
        fandoms_da_sf_test[fandom] = fandom_info
        da_sf_test_count += len(fandom_info['ids'])
        if da_sf_test_count >= da_sf_test_size:
            break
    
    # remove DA-SF test fandoms from DA-SF train fandoms
    for fandom in fandoms_da_sf_test.keys():
        if fandom in fandoms_da_sf_train:
            del fandoms_da_sf_train[fandom]

    # pull authors from DA-SF train fandoms
    for fandom, fandom_info in fandoms_da_sf_train.items():
        for a in fandom_info['authors']:
            authors_da_sf_train[a] = 1
    
    print("[open_set_unseen_fandoms] Populated %d out of %d DA-SF test examples " \
            % (da_sf_test_count, da_sf_test_size))
    extra = da_sf_test_count - da_sf_test_size
    da_sf_train_size -= extra
    da_sf_test_size = da_sf_test_count

    print("[open_set_unseen_fandoms] #fandoms in DA-SF train group ", len(fandoms_da_sf_train))
    print("[open_set_unseen_fandoms] #fandoms in DA-SF test group ", len(fandoms_da_sf_test))
    print("[open_set_unseen_fandoms] overlapping fandoms ", \
            len(fandoms_da_sf_train.keys() & fandoms_da_sf_test.keys()))
    print("[open_set_unseen_fandoms] #authors in DA-SF train group ", len(authors_da_sf_train))
    print("[open_set_unseen_fandoms] #authors in DA-SF test group ", len(authors_da_sf_test))
    print("[open_set_unseen_fandoms] overlapping authors ", \
            len(authors_da_sf_train.keys() & authors_da_sf_test.keys()))
    print("[open_set_unseen_fandoms] =======================================================")
    

    # create SA fandoms
    # fandoms_sa_train = defaultdict(dict)
    # for idx, example in enumerate(same_author_examples):
    #     if idx % 10000 == 0:
    #         print("[open_set_unseen_fandoms] processed %d SA examples" % (idx))
    #     ex_id = example['id']
    #     a1, a2 = example['authors'][0], example['authors'][1]
    #     f1, f2 = example['fandoms'][0], example['fandoms'][1]
        
    #     for a,f in zip([a1, a1], [f1, f2]):
    #         if f not in fandoms_sa_train:
    #             fandoms_sa_train[f]['ids'] = [ex_id]
    #             fandoms_sa_train[f]['authors'] = set([a])
    #         else:
    #             fandoms_sa_train[f]['ids'].append(ex_id)
    #             fandoms_sa_train[f]['authors'].add(a)

    # split SA fandoms in train and test
    # print("[open_set_unseen_fandoms] splitting SA fandoms in 2 groups ")

    # all examples belong to fandoms_sa_train
    # sort dictionary from least popular fandoms to most popular
    # move fandoms from fandoms_sa_train to fandoms_sa_test until enough SA test examples
    # least_freq_fandoms_train = sorted(fandoms_sa_train.items(), key=lambda x: len(x[1]['ids']))
    # for f, f_info in least_freq_fandoms_train:
    #     for a in f_info['authors']:
    #         authors_sa_test[a] = 1
    #     for pair_id in f_info['ids']:
    #         test_ids.append(pair_id)
    #     # move fandom info to the fandom test group
    #     fandoms_sa_test[f] = f_info
    #     sa_test_count += len(f_info['ids'])
    #     if sa_test_count >= sa_test_size:
    #         break

    # add SA examples to test whose fandoms overlap with DA-SF test
    authors_sa_train, authors_sa_test = {}, {}
    fandoms_sa_train, fandoms_sa_test = {}, {}
    sa_test_count = 0
    for idx, example in enumerate(sa_examples):
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        if f1 in fandoms_da_sf_test and f2 in fandoms_da_sf_test:
            # adding example to SA test pairs
            test_ids.append(example['id'])
            sa_test_count += 1
            # update fandoms and authors stats
            fandoms_sa_test[f1] = 1
            fandoms_sa_test[f2] = 1
            authors_sa_test[a1] = 1
            # if sa_test_count == sa_test_size:
            #     break

    # update counts
    print("[open_set_unseen_fandoms] Populated %d out of %d SA test examples " % (sa_test_count, sa_test_size))
    extra = sa_test_count - sa_test_size
    sa_train_size -= extra
    sa_test_size = sa_test_count
    test_size_so_far = sa_test_size + da_sf_test_size
    assert len(test_ids) == test_size_so_far, \
          "len(test_ids) = %d, test_size_so_far = %d" % (len(test_ids), test_size_so_far)

    # create author and fandom SA train stats
    for idx, example in enumerate(sa_examples):
        if example['id'] in test_ids:
            continue
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        fandoms_sa_train[f1] = 1
        fandoms_sa_train[f2] = 1
        authors_sa_train[a1] = 1

    # remove test fandoms from train fandoms group
    # for f in fandoms_sa_test.keys():
    #     if f in fandoms_sa_train:
    #         del fandoms_sa_train[f]

    # pull authors from train fandom
    # for f, f_info in fandoms_sa_train.items():
    #     for a in f_info['authors']:
    #         authors_sa_train[a] = 1

    print("[open_set_unseen_fandoms] #fandoms in SA train group ", len(fandoms_sa_train))
    print("[open_set_unseen_fandoms] #fandoms in SA test group ", len(fandoms_sa_test))
    print("[open_set_unseen_fandoms] overlapping fandoms ", \
            len(fandoms_sa_train.keys() & fandoms_sa_test.keys()))
    print("[open_set_unseen_fandoms] #authors in SA train group ", len(authors_sa_train))
    print("[open_set_unseen_fandoms] #authors in SA test group ", len(authors_sa_test))
    print("[open_set_unseen_fandoms] overlapping authors ", \
            len(authors_sa_train.keys() & authors_sa_test.keys()))

    print("[open_set_unseen_fandoms] =======================================================")
    print("[open_set_unseen_fandoms] adding DA-DF pairs to test set")
    da_test_count = 0
    fandoms_da_df_train, fandoms_da_df_test = {}, {}
    authors_da_df_train, authors_da_df_test = {}, {}
    # update fandoms statistics for DA pairs
    # for ex in da_sf_examples:
    #     fandoms_da_sf[ex['fandoms'][0]] = 1
    # for ex in da_df_examples:
    #     fandoms_da_df[ex['fandoms'][0]] = 1
    #     fandoms_da_df[ex['fandoms'][1]] = 1
    
    # print("[open_set_unseen_fandoms] #fandoms in DA-SF pairs", len(fandoms_da_sf.keys()))
    # print("[open_set_unseen_fandoms] #fandoms in DA-DF pairs", len(fandoms_da_df.keys()))
    # print("[open_set_unseen_fandoms] #inters(DA-SF, DA-DF)", len(fandoms_da_sf.keys()&fandoms_da_df.keys()))
    # print("[open_set_unseen_fandoms] DA-SF = %d, DA-DF = %d" % (len(da_sf_examples), len(da_df_examples)))
    #random.shuffle(da_examples)

    # add DA-DF examples to test set
    da_df_test_count = 0
    for idx, example in enumerate(da_df_examples):
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        if f1 in fandoms_da_sf_test and f2 in fandoms_da_sf_test:
            # adding example to DA-DF test pairs
            test_ids.append(example['id'])
            da_df_test_count += 1
            fandoms_da_df_test[f1], fandoms_da_df_test[f2] = 1, 1
            authors_da_df_test[a1], authors_da_df_test[a2] = 1, 1
            #if da_test_count == da_test_size:
            #    break

    # create DA-DF train fandoms and train authors stats
    for idx, example in enumerate(da_df_examples):
        if example['id'] in test_ids:
            continue
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        fandoms_da_df_train[f1] = 1
        fandoms_da_df_train[f2] = 1
        authors_da_df_train[a1] = 1

    print("[open_set_unseen_fandoms] Populated %d out of %d DA-SF test examples " \
          % (da_df_test_count, da_df_test_size))

    print("[open_set_unseen_fandoms] #fandoms in DA-DF train group ", len(fandoms_da_df_train))
    print("[open_set_unseen_fandoms] #fandoms in DA-DF test group ", len(fandoms_da_df_test))
    print("[open_set_unseen_fandoms] overlapping fandoms ", \
            len(fandoms_da_df_train.keys() & fandoms_da_df_test.keys()))
    print("[open_set_unseen_fandoms] #authors in DA-DF train group ", len(authors_da_df_train))
    print("[open_set_unseen_fandoms] #authors in DA-DF test group ", len(authors_da_df_test))
    print("[open_set_unseen_fandoms] overlapping authors ", \
            len(authors_da_df_train.keys() & authors_da_df_test.keys()))
      
    test_ids_map = {test_id:1 for test_id in test_ids}
    train_ids = [example['id'] for example in examples if example['id'] not in test_ids_map]
    train_ids_map = {train_id:1 for train_id in train_ids}

    # statistics
    train_stats = defaultdict(int)
    test_stats = defaultdict(int)
    for example in examples:
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        stats_dict = train_stats if example['id'] in train_ids_map else test_stats
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
    
    for split_name, stats_dict in zip(["TRAIN", "TEST"], [train_stats, test_stats]):
        split_ids = train_ids if split_name == 'TRAIN' else test_ids
        print("%s size: %d" % (split_name, len(split_ids)))
        print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
        print("        Same fandom pairs: ", stats_dict['sa_sf'])
        print("        Different fandom pairs: ", stats_dict['sa_df'])
        print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
        print("        Same fandom pairs: ", stats_dict['da_sf'])
        print("        Different fandom pairs: ", stats_dict['da_df'])

        

def make_two_author_groups(authors_source: Union[str, Dict]) -> (Dict, Dict):
    """
    Algorithm 1 from Boenninghoff et al. 2020
    Splits a list of authors and their documents into 2 groups:
     - a group of authors with only 1 document
     - a group of authors with 2 ore more documents (even number)
    Arguments:
        authors_source: either a ```str``` indicating the path to an author folder or a ```Dict```
                        with author data read from the author folder. An author folder
                        contains .jsonl files, one per each author and is created via 
                        ```get_author_docs_from_jsonl```. The author data is a dictionary 
                        read from the abovementioned folder using ```get_author_data_from_folder```
    """
    single_doc_authors = {}
    even_doc_authors = {}
    if type(authors_source) == 'str':
        for idx, author_file in enumerate(os.listdir(authors_source)):
            if idx % 1000 == 0:
                print("[make_two_author_groups] processed ", idx, " authors")
            author_id = author_file[:-6]
            with open(os.path.join(authors_source, author_file)) as fp:
                author_data = []
                for line in fp:
                    author_data.append(json.loads(line))
                
                assert len(author_data) > 0, "error in loading %s" % author_file
                
                if len(author_data) == 1:
                    single_doc_authors[author_id] = author_data
                elif len(author_data) % 2 == 0:
                    even_doc_authors[author_id] = author_data
                else:
                    single_doc_authors[author_id] = [author_data[0]]
                    even_doc_authors[author_id] = author_data[1:]
    elif type(authors_source) == defaultdict or type(authors_source) == Dict:
        for author_id, author_data in authors_source.items():
            if len(author_data) == 1:
                single_doc_authors[author_id] = author_data
            elif len(author_data) % 2 == 0:
                even_doc_authors[author_id] = author_data
            else:
                single_doc_authors[author_id] = [author_data[0]]
                even_doc_authors[author_id] = author_data[1:]

    for author, docs in single_doc_authors.items():
        assert len(docs) == 1, "single-doc author group has more than one document"

    for author, docs in even_doc_authors.items():
        assert len(docs) > 0 and len(docs) % 2 == 0, "even-doc author group has odd number of documents" 
    
    return single_doc_authors, even_doc_authors


def clean_after_sampling(author_id: str, 
                         author_docs: List[Dict], 
                         single_doc_authors: Dict[str, List], 
                         even_doc_authors: Dict[str, List]):
    """
    Algorithm 2 from from Boenninghoff et al. 2020

    Args:
        author_id (str): [description]
        author_docs (List[Dict]): [description]
        single_doc_authors (Dict[List]): [description]
        even_doc_authors (Dict[List]): [description]

    Returns:
        a new example (optional) and the two updated author groups
    """
    new_example = None
    if len(author_docs) > 1:
        even_doc_authors[author_id] = author_docs
    elif len(author_docs) == 1:
        doc = author_docs[0]
        f1 = doc['fandom']
        d1 = doc['text']
        # check if author in single-doc author group
        if author_id in single_doc_authors:
            doc = single_doc_authors[author_id][0]
            f2 = doc['fandom']
            d2 = doc['text']
            new_example = {
                "same": True,
                "authors": [author_id, author_id],
                "fandoms": [f1, f2],
                #"pair": [d1, d2],
            }
            del single_doc_authors[author_id]
        else:
            single_doc_authors[author_id] = author_docs

    return new_example, single_doc_authors, even_doc_authors


def sample_pairs(authors_data: Dict, output_folder: str):
    """
    Algorithm 3 from Boenninghoff et al. 2020
    Returns same-author pairs as well as different-author pairs from the given
    authors data.

    Args:
        authors_data (Dict): authors data
        output_folder (str): [description]
    """
    if not os.path.exists(output_folder):
        print("[sample_pairs] folder %s doesn't exist, trying to create it" % (output_folder))
        os.mkdir(output_folder)

    single_doc_authors, even_doc_authors = make_two_author_groups(authors_data)
    samples_count = 0
    while len(single_doc_authors) > 1 and len(even_doc_authors) > 0:
        if samples_count % 100 == 0:
            print("[sample_pairs] samples_count = ", samples_count)
        # sample same-author pair
        if len(even_doc_authors) > 0:
            # sample author
            author_id = random.choice(list(even_doc_authors.keys()))
            author_docs = even_doc_authors[author_id]
            del even_doc_authors[author_id]

            # sample two documents from author's documents
            same_docs = random.sample(author_docs, k=2)
            f1, f2 = same_docs[0]['fandom'], same_docs[1]['fandom']
            d1, d2 = same_docs[1]['text'], same_docs[1]['text']

            # create same-author pair
            example = {
                "same": True, "authors": [author_id, author_id], "fandoms": [f1, f2]#, 
                #"pair": [d1, d2]
            }
            with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                json.dump(example, fp)
                fp.write('\n')
                samples_count += 1

            # remove sampled documents
            author_docs = [doc for doc in author_docs if doc not in same_docs]

            # add remaining docs to even-docs author group
            example, single_doc_authors, even_doc_authors = clean_after_sampling(
                author_id, author_docs, single_doc_authors, even_doc_authors
            )
            if example:
                with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                    json.dump(example, fp)
                    fp.write('\n')
                    samples_count += 1

        # sample different-author pair
        if len(even_doc_authors) > 1:
            # sample two authors
            author_ids = random.sample(list(even_doc_authors.keys()), k=2)
            fst_author_id, snd_author_id = author_ids[0], author_ids[1]
            fst_docs, snd_docs = even_doc_authors[fst_author_id], even_doc_authors[snd_author_id]
            
            # remove authors from group
            del even_doc_authors[fst_author_id]
            del even_doc_authors[snd_author_id]
            
            fst_fandoms = set([doc['fandom'] for doc in fst_docs])
            snd_fandoms = set([doc['fandom'] for doc in snd_docs])
            common_fandoms = list(fst_fandoms.intersection(snd_fandoms))
            #print("Common fandoms = ", common_fandoms)

            if len(common_fandoms) > 0:
                # try to sample same-fandom pair
                f = random.choice(common_fandoms)
                fst_docs_population = [doc for doc in fst_docs if doc['fandom'] == f]
                snd_docs_population = [doc for doc in snd_docs if doc['fandom'] == f]
            else:
                fst_docs_population = fst_docs
                snd_docs_population = snd_docs

            # take random doc from 1st author and random doc from 2nd author
            fst_doc = random.choice(fst_docs_population)
            snd_doc = random.choice(snd_docs_population)

            # remove documents from each author
            fst_docs = [doc for doc in fst_docs if doc != fst_doc]
            snd_docs = [doc for doc in snd_docs if doc != snd_doc]

            # create example
            example = {
                "same": False,
                "authors": [fst_author_id, snd_author_id],
                "fandoms": [fst_doc["fandom"], snd_doc["fandom"]]#,
                #"pair": [fst_doc["text"], snd_doc["text"]]
            }
            with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                json.dump(example, fp)
                fp.write('\n')
                samples_count += 1
            
            # add authors back to their groups
            for (author_id, author_docs) in zip(author_ids, [fst_docs, snd_docs]):
                example, single_doc_authors, even_doc_authors = clean_after_sampling(
                    author_id, author_docs, single_doc_authors, even_doc_authors
                )
                if example:
                    with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                        json.dump(example, fp)
                        fp.write('\n')
                        samples_count += 1
        elif len(single_doc_authors) > 1:
            # sample two authors
            author_ids = random.sample(list(single_doc_authors.keys()), k=2)
            fst_author_id, snd_author_id = author_ids[0], author_ids[1]

            fst_doc, snd_doc = single_doc_authors[fst_author_id][0], single_doc_authors[snd_author_id][0]
            del single_doc_authors[fst_author_id]
            del single_doc_authors[snd_author_id]

            example = {
                "same": False,
                "authors": [fst_author_id, snd_author_id],
                "fandoms": [fst_doc["fandom"], snd_doc["fandom"]]#,
                #"pair": [fst_doc["text"], snd_doc["text"]]
            }
            with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                json.dump(example, fp)
                fp.write('\n')
                samples_count += 1


def split_jsonl_dataset(path_to_original_jsonl: str,
                        path_to_train_jsonl: str, 
                        path_to_test_jsonl: str,
                        split_function: Callable[[List[Dict], float], Tuple[List]],
                        test_split_percentage: float):
    """
    Split the PAN dataset into 2 splits. This wrapper function can be used to split the original dataset
    into train and test, as well as further split the train set into train and val.
    Args:
        path_to_original_jsonl (str): path to existing .jsonl file, such as ```pan20-av-large-no-text.jsonl```
        path_to_train_jsonl (str): path to .jsonl file where the training examples will be saved
        path_to_test_jsonl (str): path to .jsonl file where the test examples will be saved
        split_function (Callable): split function to be used (split_pan_dataset_closed_set_v1 or
                                   split_pan_dataset_closed_set_v2)
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    """
    if os.path.exists(path_to_original_jsonl):
        examples = read_jsonl_examples(path_to_original_jsonl)
    else:
        print("File %s doesn't exist" % (path_to_original_jsonl))

    # split into train and test
    train_ids, test_ids = split_function(
        examples=examples, 
        test_split_percentage=test_split_percentage
    )

    # saving examples to train and test .jsonl files
    print("Writing examples to %s and %s" % (path_to_train_jsonl, path_to_test_jsonl))
    with open(path_to_train_jsonl, "w") as f, open(path_to_test_jsonl, "w") as g:
        for idx, example in enumerate(examples):
            if idx % 10000 == 0:
                print("[split_jsonl_dataset] Wrote %d examples" % (idx))
            if example['id'] in test_ids:
                json.dump(example, g)
                g.write('\n')
            else:
                json.dump(example, f)
                f.write('\n')


def print_dataset_statistics(examples: List[Dict]):
    stats_dict = defaultdict(int)
    authors_dict = {}
    for example in examples:
        authors_dict[example['authors'][0]] = 1
        authors_dict[example['authors'][1]] = 1
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
        
    print("Dataset size: ", len(examples))
    print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
    print("        Same fandom pairs: ", stats_dict['sa_sf'])
    print("        Different fandom pairs: ", stats_dict['sa_df'])
    print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
    print("        Same fandom pairs: ", stats_dict['da_sf'])
    print("        Different fandom pairs: ", stats_dict['da_df'])
    print("Number of unique authors: ", len(authors_dict))


if __name__ == '__main__':
    remote_xl_paths = {
        "data": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large.jsonl",
        "gt": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large-truth.jsonl",
        "original": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large.jsonl",
        "no_test": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-notest.jsonl",
        "train": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-train.jsonl",
        "val": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-val.jsonl",
        "test": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-test.jsonl"
    }

    remote_xs_paths = {
        "data": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl",
        "gt": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl",
        "original": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small.jsonl",
        "no_test": "/pan2020/pan20-authorship-verification-training-large/pan20-av-small-notest.jsonl",
        "train": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-train.jsonl",
        "val": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-val.jsonl",
        "test": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-test.jsonl"
    }

    local_xl_paths = {
        "original": "../data/pan2020_xl/pan20-av-large.jsonl",
        "no_test": "../data/pan2020_xl/pan20-av-large-notest.jsonl",
        "train": "../data/pan2020_xl/pan20-av-large-train.jsonl",
        "val": "../data/pan2020_xl/pan20-av-large-val.jsonl",
        "test": "../data/pan2020_xl/pan20-av-large-test.jsonl"
    }

    local_xs_paths = {
        "original": "../data/pan2020_xs/pan20-av-small.jsonl",
        "no_test": "../data/pan2020_xs/pan20-av-small-notest.jsonl",
        "train": "../data/pan2020_xs/pan20-av-small-train.jsonl",
        "val": "../data/pan2020_xs/pan20-av-small-val.jsonl",
        "test": "../data/pan2020_xs/pan20-av-small-test.jsonl"
    }
    
    # TODO: change with the appropriate paths
    paths_dict = local_xl_paths

    # Split original dataset into:
    #   - Train (pan20-av-*-notest.jsonl)
    #   - Test (pan20-av-*-test.jsonl)
    # We already did this once so we should keep the Test set intact
    # split_jsonl_dataset(
    #     path_to_original_jsonl=paths_dict['original'],
    #     path_to_train_jsonl=paths_dict['no_test'],
    #     path_to_test_jsonl=paths_dict['test'],
    #     split_function=split_pan_dataset_closed_set_v1,
    #     test_split_percentage=0.05
    # )

    # split Train dataset into Train and Val
    # split_jsonl_dataset(
    #     path_to_original_jsonl=paths_dict['no_test'],
    #     path_to_train_jsonl=paths_dict['train'],
    #     path_to_test_jsonl=paths_dict['val'],
    #     split_function=split_pan_dataset_closed_set_v2,
    #     test_split_percentage=0.05
    # )

    # test_examples = read_jsonl_examples('../data/pan2020_xl/backup/xl/v2_split/pan20-av-large-test.jsonl')
    # test_authors = {}
    # for example in test_examples:
    #     a1 = example['authors'][0]
    #     a2 = example['authors'][1]
    #     test_authors[a1] = 1
    #     test_authors[a2] = 1

    train_authors = {}
    #train_examples = read_jsonl_examples('../data/pan2020_xl/backup/xl/v2_split/pan20-av-large-test.jsonl')
    train_examples = read_jsonl_examples('../data/pan2020_xl/pan20-av-large.jsonl')
    split_pan_dataset_open_set_unseen_fandoms(train_examples, 0.05)

    # train_set = set(train_authors)
    # test_set = set(test_authors)
    # print("Number of authors in train set: ", len(train_set))
    # print("Number of authors in test set: ", len(test_set))
    # print("Number of overlapping authors: ", len(test_set.intersection(train_set)))