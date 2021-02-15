import json
import os
import random
import numpy as np
from collections import defaultdict
from typing import Dict, Union, List

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



def split_pan_dataset_old(data_path: str, 
                          percentage: float, 
                          dataset_size: str = ['small', 'large']) -> (List, List):
    # DA pairs
    # Algorithm 1: slow
    # (A1, A2) - move this to test, because A1 exists in other DA pairs, as well as A2
    # (A1, A3) - keep this pair
    # (A2, A5) 
    # (A2, A7)
    # for (ai, aj) in DA pairs:
    #   if ai in other DA pairs and aj in other DA pairs:
    #       move (ai, aj) to test
    #       mark pairs in which ai and aj appear as train    
    # all pairs belong to train
    is_test = [False] * len(diff_author_examples)
    is_locked = [False] * len(diff_author_examples) #examples cannot be switched 
    fst_authors = [ex['authors'][0] for ex in diff_author_examples]
    snd_authors = [ex['authors'][1] for ex in diff_author_examples]
    for idx, da_pair in enumerate(diff_author_examples):
        print("[split_pan_dataset] id = ", idx)
        if is_locked[idx] or is_test[idx]:
            continue
        a1, a2 = da_pair['authors'][0], da_pair['authors'][1]
        
        # search author a1 in other pairs marked as training
        a1_idx = -1
        for i, author in enumerate(fst_authors):
            if is_test[i]:
                continue
            if a1 == author:
                a1_idx = i

        # search author a2 in other pairs marked as training
        a2_idx = -1
        for i, author in enumerate(snd_authors):
            if is_test[i]:
                continue
            if a2 == author:
                a2_idx = i        

        # if we found a1 and a2 in other training pairs, we can safely mark
        # (a1, a2) for testing, as well as lock pairs (a1_idx, *) and (*, a2_idx)
        # for training
        if a1_idx >= 0 and a2_idx >= 0:
            is_test[idx] = True
            is_locked[idx] = True
            is_locked[a1_idx] = True
            is_locked[a2_idx] = True

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

def pan_dataset_closed_train_test_split(examples: List[Dict], 
                                        test_split_percentage: float) -> (List, List):
    """
    Split PAN 2020 dataset in Train and Test under the closed-set assumption. This requires that
    authors in Test set appear in Train as well. However, due to the large number of authors 
    in the different-author (DA) pairs, it is difficult to achieve strictly. We try to guarantee 
    that at least one of the authors (ai, aj) in DA Test pairs appears in DA Train Pairs or
    in same-author (SA) Train pairs.

    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    """
    assert test_split_percentage > 0 and test_split_percentage < 1, "test size in (0,1)"
    sizes = {
        "small": {"size": 52601, "positive": 27834, "negative": 24767},
        "large": {"size": 275565, "positive": 147778, "negative": 127787}
    }

    # determine Train/Test sizes
    same_author_examples = [ex for ex in examples if ex['same']]
    diff_author_examples = [ex for ex in examples if not ex['same']]
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
    for example in diff_author_examples:
        fst_author_id = example['authors'][0] # a1
        snd_author_id = example['authors'][1] # a2

        # check if a1 or a2 appear in other DA pairs
        fst_frequent = diff_author_freq[fst_author_id] >= 2
        snd_frequent = diff_author_freq[snd_author_id] >= 2
        if fst_frequent or snd_frequent:
            test_ids.append(example['id'])
            if fst_frequent:
                test_author_ids.add(fst_author_id)
                diff_author_freq[fst_author_id] -= 1
            if snd_frequent:
                test_author_ids.add(snd_author_id)
                diff_author_freq[snd_author_id] -= 1
        # check if a1 or a2 appear in SA pairs
        elif fst_author_id in sa_authors_ids or snd_author_id in sa_authors_ids:
            test_ids.append(example['id'])
            if fst_author_id in sa_authors_ids:
                test_author_ids.add(fst_author_id)
            if snd_author_id in sa_authors_ids:
                test_author_ids.add(snd_author_id)

    da_ids = [example['id'] for example in diff_author_examples]
    train_ids = [ex_id for ex_id in da_ids if ex_id not in test_ids]

    print("Number of different-author (DA) pairs: ", len(diff_author_examples))
    print(" Number of candidate DA test pairs: ", len(test_ids))
    print(" Number of candidate DA train pairs: ", len(train_ids))

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
    train_ids = []
    for idx, example in enumerate(examples):
        if example['id'] not in test_ids_map:
            train_ids.append(example['id'])
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

def split_jsonl_into_json_folder(path_to_jsonl: str, pan_authors_folder: str):
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
                        test_split_percentage: float):
    """
    Split the PAN dataset into 2 splits.
    Args:
        path_to_original_jsonl (str): path to existing .jsonl file, such as ```pan20-av-large-no-text.jsonl```
        path_to_train_jsonl (str): path to .jsonl file where the training examples will be saved
        path_to_test_jsonl (str): path to .jsonl file where the test examples will be saved
    """
    if os.path.exists(path_to_original_jsonl):
        examples = read_jsonl_examples(path_to_original_jsonl)
    else:
        print("File %s doesn't exist" % (path_to_original_jsonl))

    # split into train and test
    train_ids, test_ids = pan_dataset_closed_train_test_split(
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
    #split_large_jsonl(PAN2020_SMALL_VAL, "/pan2020/pan20-authorship-verification-training-small/val")
    # sanity checks
    # write_jsonl_to_folder("../data/pan20-av-large.jsonl", "../data/pan20-av-large")

    # write author data to folder
    # split_jsonl_into_json_folder("../data/pan20-av-large.jsonl", "../data/pan20-av-large-authors")

    #authors_data = get_authors_data_from_folder("../data/pan20-av-large-authors")#, num_authors=100000)
    #sample_pairs(authors_data=authors_data, output_folder="../data/pan20-av-large_sampled")
    
    # load examples
    #examples = read_jsonl_examples("../data/pan20-av-large-no-text.jsonl")

    # split into train and test
    # train_ids, test_ids = pan_dataset_closed_train_test_split(
    #     examples=examples, 
    #     test_split_percentage=0.05
    # )

    split_jsonl_dataset(
        path_to_original_jsonl="../data/pan20-av-large.jsonl",
        path_to_train_jsonl="../data/pan20-av-large-train.jsonl",
        path_to_test_jsonl="../data/pan20-av-large-test.jsonl",
        test_split_percentage=0.05
    )

    # author_docs_len = [len(docs) for docs in authors_data.values()]
    # print("Total documents = ", sum(author_docs_len))

    # sa_sf, sa_df, da_sf, da_df = 0, 0, 0, 0
    # root_dir = "../data/pan20-av-large_sampled"
    # for fname in os.listdir(root_dir):
    #     with open(os.path.join(root_dir, fname)) as fp:
    #         entry = json.load(fp)
    #         if entry['same']:
    #             if entry['fandoms'][0] == entry['fandoms'][1]:
    #                 sa_sf += 1
    #             else:
    #                 sa_df += 1
    #         else:
    #             if entry['fandoms'][0] == entry['fandoms'][1]:
    #                 da_sf += 1
    #             else:
    #                 da_df += 1
    
    # print("Total examples: ", sa_sf+sa_df+da_sf+da_df)
    # print("SA examples: ", sa_sf+sa_df)
    # print("     SA examples, same fandom: ", sa_sf)
    # print("     SA examples, diff fandom: ", sa_df)
    # print("DA examples: ", da_sf+da_df)
    # print("     DA examples, same fandom: ", da_sf)
    # print("     DA examples, diff fandom: ", da_df)

    # train_files = os.listdir('../data/train')
    # val_files = os.listdir('../data/val')
    # print("# train files = ", len(train_files))
    # print("# val files = ", len(val_files))
    # print("train+val files = ", len(train_files) + len(val_files))

    # with open("../data/pan20-av-large-no-text.jsonl") as f:
    #     count = 0
    #     for line in f:
    #         count += 1
    # print("json size = ", count)
    # statistics
    # sa_sf, sa_df, da_sf, da_df = 0, 0, 0, 0
    # da_authors = defaultdict(int)
    # da_count = set()
    # sa_count = set()
    # da_pairs = 0
    # with open("../data/pan20-av-large-no-text.jsonl") as f:
    #     for idx, line in enumerate(f):
    #         entry = json.loads(line)
    #         a1 = int(entry['authors'][0])
    #         a2 = int(entry['authors'][1])
    #         f1 = entry['fandoms'][0]
    #         f2 = entry['fandoms'][1]
    #         if entry['same']:    
    #             sa_count.add(a1)
    #             sa_count.add(a2)
    #         else:
    #             if a1 > a2:
    #                 a1, a2 = a2, a1
    #             da_pairs += 1
    #             da_authors[a1] += 1
    #             da_authors[a2] += 1
                
    #             da_count.add(a1)
    #             da_count.add(a2)


                
