

# PAN 2020 Dataset statistics


## PAN 2020 XL statistics
|  | same fandom | cross-fandom |
|--|-------------|--------------|
|same-author pairs| 0 | 147.778 |
|different-author pairs| 23.131 | 104.656 |

 - same-author pairs are constructed from 41.370 authors, while different-author pairs are constructed from 251.503 authors
 - 14.704 authors in SA pairs can be found in DA pairs as well
 - 3.966 authors in DA pairs appear in at least one DA pair
 - author tuples (Ai, Aj) in DA pairs are unique (i.e. authors 532 and 7145 can be found in this combination only once in DA pairs)
 - there are 494.236 distinct documents

We now detail the closed-set and open-set setups. In both setups, we split the XL dataset into 95% training and 5% test and the XS dataset into 90% training and 10% test. 

## Closed-set setup
In the closed-set setup authors of same-author pairs in the validation/test set are guaranteed to appear in the training set. However, this is difficult to achieve for the different-author pairs of the PAN 2020 dataset, as they span a large number of authors with few occurences each.

### Files
Download [```pan2020.zip```](https://drive.google.com/file/d/18UPhYsdtFa8ObD0M6AeMdLxJ1XH42vHQ/view?usp=sharing) and unzip it. This is the structure of its content: 
```
xl/
   v1_split/
            pan20-av-large-test.jsonl
            pan20-av-large-notest.jsonl
   v2_split/
            pan20-av-large-test.jsonl
            pan20-av-large-notest.jsonl
xs/
   v1_split/
            pan20-av-small-test.jsonl
            pan20-av-small-notest.jsonl
   v2_split/
            pan20-av-small-test.jsonl
            pan20-av-small-notest.jsonl
```
We try two variants of splitting the datasets, called ```v1``` and ```v2```. The splits for the PAN 2020 large dataset can be found in the ```xl``` folder, while the splits for the PAN 2020 small dataset can be found in the ```xs``` folder.


### Version v1
In this version, authors of same-author pairs in the validation set are guaranteed to appear in the training set, while some authors of different-author pairs in the validation set may not appear in the training set.

Here are some dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF | 
|--------|----------|------|----|----|
|PAN 2020 large - original| ```pan20-av-large.jsonl``` | 275565 | - | - | -|
|PAN 2020 large - test | ```pan20-av-large-test.jsonl``` | 13784 | 7395/0/7395 | 6389/1114/5275 |
|PAN 2020 large - w/o test| ```pan20-av-large-no-test.jsonl``` | 261784 | - | - | -|
|PAN 2020 large - train | ```pan20-av-large-train.jsonl``` | 248688 | 133359/0/133359 | 115329/20945/94384 |
|PAN 2020 large - val | ```pan20-av-large-val.jsonl``` | 13090 | 7024/0/7024 | 6069/1072/4997 |  
where:
 - SA: same-author pairs
 - SA-SF: same-author pairs that have the same fandom
 - SA-DF: same-author pairs that have different fandoms
 - DA: different-author pairs
 - DA-SF: different-author pairs that have the same fandom
 - DA-DF: different-author pairs that have different fandoms


To split the PAN 2020 large dataset (```pan20-av-large-notest.jsonl```) into train (```pan20-av-large-train.jsonl```) and validation (```pan20-av-large-val.jsonl```) splits using the ```v1``` version, call the ```split_jsonl_dataset``` function in ```preprocess/split_train_val.py```:
```
cd preprocess
python split_train_val.py
```
Make sure to specify the correct paths:
 - ```path_to_train_jsonl``` is where you want to save your training split
 - ```path_to_test_jsonl``` is where you want to save your validation split
```
    # split Train dataset into Train and Val
    split_jsonl_dataset(
        path_to_original_jsonl=pan20-av-notest.jsonl,
        path_to_train_jsonl=pan20-av-large-train.jsonl,
        path_to_test_jsonl=pan20-av-large-val.jsonl,
        split_function=split_pan_dataset_closed_set_v1,
        test_split_percentage=0.05
    )
```


Different-author pairs:
 - the DA pairs are randomly assigned to train/val split
 - unseen authors can appear at evaluation, for instance (A1, A2) in training set and (A3, A4) in val set. 

 Same-author (SA) pairs:
 - while populating the validation split, SA pairs are evenly assigned to train/val splits
 - for instance, if we have 10 SA examples from a given author, we assign 5 examples to training split and 5 examples to validation split. This ensures that the author of SA pairs in the validation split has been 'seen' at training time*. 
 - *this may result in unseen fandoms at validation time though, for instance (A1, F1, A1, F2) at training time and (A1, F3, A1, F4) at validation 

 

### Version v2
If we separate the DA pairs (ai, aj) into two groups Train and Test, such that both authors (ai, aj) of DA pairs in Test also appear in DA pairs in Train
(or SA pairs), we get the following stats:
 - Number of DA pairs:  127787
 - Number of candidate test pairs:  181
 - Number of candidate train pairs:  127606

The small number of candidate test pairs suggest that most of the authors in the DA pairs of the test split are 'unseen' at training
time. To loosen this restriction, we can split the DA pairs such that at least one of the authors (ai, aj) in a DA Test pair appears in
other DA Train pairs or in SA pairs. We get the following stats:
 - Number of DA pairs:  127787
 - Number of candidate test pairs:  17894
 - Number of candidate train pairs:  109893

We therefore split a PAN dataset into Train and Val/Test such that at least one of the authors in DA Test pairs appears DA train pairs or SA train pairs.
The SA pairs of an author A are equally distributed between Train and Test.

Here are some dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF | 
|--------|----------|------|----|----|
|PAN 2020 large - original| ```pan20-av-large.jsonl``` | 275565 | - | - | -|
|PAN 2020 large - test | ```pan20-av-large-test.jsonl``` | 13785 | 7396/0/7396 | 6389/355/6034 |
|PAN 2020 large - w/o test| ```pan20-av-large-no-test.jsonl``` | 261784 | - | - | -|
|PAN 2020 large - train | ```pan20-av-large-train.jsonl``` | 248688 | 133359/0/133359 | 115329/22420/92909 |
|PAN 2020 large - val | ```pan20-av-large-val.jsonl``` | 13090 | 7023/0/7023 | 6069/356/5713 |  

To split the PAN 2020 large dataset (```pan20-av-large-notest.jsonl```) into train and validation splits using the ```v2``` version, call the ```split_jsonl_dataset``` function in ```preprocess/split_train_val.py```:
```
cd preprocess
python split_train_val.py
```
Make sure to specify the correct paths:
```
    # split Train dataset into Train and Val
    split_jsonl_dataset(
        path_to_original_jsonl=pan20-av-notest.jsonl,
        path_to_train_jsonl=pan20-av-large-train.jsonl,
        path_to_test_jsonl=pan20-av-large-val,
        split_function=split_pan_dataset_closed_set_v2,
        test_split_percentage=0.05
    )
```


## Open-set setup
We make sure that authors or fandoms in the training split do no appear in the validation split.

| Dataset      | Number of pairs | Positive | Negative
| ----------- | ----------- | ---------| --------|
| PAN 2020 XL | 275565 | 147778 | 127787 |
| PAN 2020 XL train | TODO | TODO | TODO |
| PAN 2020 XL val | TODO | TODO | TODO |
| PAN 2020 XS | 52601 | 27834 | 24767 |
| PAN 2020 XS train | TODO | TODO | TODO |
| PAN 2020 XS val | TODO | TODO | TODO |


## Datasets
| Dataset      | Number of pairs | Positive | Negative
| ----------- | ----------- | ---------| --------|
| PAN 2020 XL | 275565 | 147778 | 127787 |
| PAN 2020 XS | 52601 | 27834 | 24767 |

### PAN 2020 XS (Small dataset)
 Data file: ```pan20-authorship-verification-training-small.jsonl```

 Ground truth file: ```pan20-authorship-verification-training-small-truth.jsonl```

 We concatenate the data and ground truth files into a single file ```pan20-av-small.jsonl``` by calling the ```merge_data_and_labels()``` function.

### PAN 2020 XL (Large dataset)
 Data file: ```pan20-authorship-verification-training-large.jsonl```

 Ground truth file: ```pan20-authorship-verification-training-large-truth.jsonl```

 We concatenate the data and ground truth files into a single file ```pan20-av-large.jsonl``` by calling the ```merge_data_and_labels()``` function.

  ### Train/validation splits
 We split ```pan20-av-large.jsonl``` into a 95% training set and a 5% validation set using the ```split_train_val()``` function.

 Training file: ```pan20-av-large-train.jsonl```
 Validation file: ```pan20-av-large-val.jsonl```


### Storing examples in separate ```.json``` files
 Since the ```.jsonl``` files are quite large, we use the ```split_large_jsonl()``` function to store examples from ```pan20-av-large-*.jsonl``` into separate ```.json``` files.