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

def split_pan_dataset(data_path: str, 
                      percentage: float, 
                      dataset_size: str = ['small', 'large']) -> (List, List):
    """
    Split PAN 2020 dataset in 2 splits and S1 and S2 such that authors and fandoms
    in S2 appear in S1 as well. This function can be used to split the initial
    dataset into train and test, as well as to split the train subset into train and dev.
    Args:
        data_path: path to .jsonl file containing the author pairs, one per line
                   It is recommended to use the ```pan20-av-*-no-text.jsonl``` file,
                   because it doesn't store the text pair and is significantly smaller.
        percentage: percentage of the smaller split 
        dataset_size: refers to the PAN 2020 large or small dataset
    """
    sizes = {
        "small": {"size": 52601, "positive": 27834, "negative": 24767},
        "large": {"size": 275565, "positive": 147778, "negative": 127787}
    }
    # load examples
    examples = []
    with open(data_path) as fp:
        for line in fp:
            examples.append(json.loads(line))

    same_author_examples = [ex for ex in examples if ex['same']]
    diff_author_examples = [ex for ex in examples if not ex['same']]
    
    # add test ids of same-author pairs
    test_ids = []
    # {'author_id': [ids of SA pairs of this authors]}
    same_author_docs = defaultdict(list)
    for example in same_author_examples:
        author_id = example['authors'][0]
        same_author_docs[author_id].append(example['id'])

    sa_test_size = int(percentage * len(same_author_examples))
    count = 0
    for author_id, pair_ids in same_author_docs.items():
        author_docs_num = len(pair_ids)
        if author_docs_num >= 2:
            test_ids += pair_ids[:author_docs_num // 2]
            count += author_docs_num // 2
        if count > sa_test_size:
            break

    # {'author_id': [ids of DA pairs in which this author appear]}
    diff_author_docs = defaultdict(list)
    for example in diff_author_examples:
        fst_author_id = example['authors'][0]
        snd_author_id = example['authors'][1]
        if int(fst_author_id) < int(snd_author_id):
            author_id = fst_author_id
        else:
            author_id = snd_author_id
        diff_author_docs[author_id].append(example['id'])

    # DA pairs
    # group them by author
    # (A1, A2) - move this to test, because A1 exists in other DA pairs, as well as A2
    # (A1, A3) - keep this pair
    # (A2, A5) 
    # (A2, A7)
    
    


    



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

def get_author_docs_from_jsonl(path_to_jsonl: str, pan_authors_folder: str):
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



#def split_train_val(author_folder: str):


# class Pipeline:
#     def __init__(self, task: str = [author_folder: str, options = ['']):
#         """
        
#         Args:
#             author_folder (str): path to folder where each author has a .jsonl file containing all its texts
#         """
#         self.author_folder = author_folder

#     def 


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
    # get_author_docs_from_jsonl("../data/pan20-av-large.jsonl", "../data/pan20-av-large-authors")

    authors_data = get_authors_data_from_folder("../data/pan20-av-large-authors")#, num_authors=100000)
    #sample_pairs(authors_data=authors_data, output_folder="../data/pan20-av-large_sampled")
    
    author_docs_len = [len(docs) for docs in authors_data.values()]
    print("Total documents = ", sum(author_docs_len))

    sa_sf, sa_df, da_sf, da_df = 0, 0, 0, 0
    root_dir = "../data/pan20-av-large_sampled"
    for fname in os.listdir(root_dir):
        with open(os.path.join(root_dir, fname)) as fp:
            entry = json.load(fp)
            if entry['same']:
                if entry['fandoms'][0] == entry['fandoms'][1]:
                    sa_sf += 1
                else:
                    sa_df += 1
            else:
                if entry['fandoms'][0] == entry['fandoms'][1]:
                    da_sf += 1
                else:
                    da_df += 1
    
    print("Total examples: ", sa_sf+sa_df+da_sf+da_df)
    print("SA examples: ", sa_sf+sa_df)
    print("     SA examples, same fandom: ", sa_sf)
    print("     SA examples, diff fandom: ", sa_df)
    print("DA examples: ", da_sf+da_df)
    print("     DA examples, same fandom: ", da_sf)
    print("     DA examples, diff fandom: ", da_df)


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


                
