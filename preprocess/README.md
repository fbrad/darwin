

# PAN 2020 Dataset statistics

## Overview

### PAN 2020 XL statistics
|  | same fandom | cross-fandom |
|--|-------------|--------------|
|same-author pairs| 0 | 147.778 |
|different-author pairs| 23.131 | 104.656 |

 - same-author pairs are constructed from 41.370 authors, while different-author pairs are constructed from 251.503 authors
 - 14.704 authors in SA pairs can be found in DA pairs as well
 - 3.966 authors in DA pairs appear in at least one DA pair
 - author tuples (Ai, Aj) in DA pairs are unique (i.e. authors 532 and 7145 can be found in this combination only once in DA pairs)
 - there are 494.236 distinct documents

If we separate the DA pairs (ai, aj) into two groups Train and Test, such that both authors (ai, aj) of DA pairs in Test also appear in DA pairs in Train
(or SA pairs), we get the following stats:
 - Number of DA pairs:  127787
 - Number of candidate test pairs:  181
 - Number of candidate train pairs:  127606
The small number of candidate test pairs suggest that most of the authors in the DA pairs of the test split are 'unseen' at training
time. To loosen this restriction, we can split the DA pairs such that at least one of the authors (ai, aj) in a DA Test pair appears in
other DA Train pairs or in a SA pair. We get the following stats:
 - Number of DA pairs:  127787
 - Number of candidate test pairs:  17894
 - Number of candidate train pairs:  109893


We now detail the closed-set and open-set setups. In both setups, we split the XL dataset into 95% training and 5% validation and the XS dataset into 90% training and 10% validation. 

### Closed-set setup (new)
In this setup authors and topics from the validation set pairs have been seen during training.

### Closed-set setup (old)
In this setup, authors of same-author pairs in the validation set are guaranteed to appear in the training set, while authors of different-author pairs in the validation set may not appear in the training set.

Different-author pairs:
 - the DA pairs are randomly assigned to train/val split*
 - unseen authors can appear at evaluation, for instance (A1, A2) in training set and (A3, A4) in val set. 

 Same-author (SA) pairs:
 - while populating the validation split, SA pairs are evenly assigned to train/val splits
 - for instance, if we have 10 SA examples from a given author, we assign 5 examples to training split and 5 examples to validation split. This ensures that the author of SA pairs in the validation split has been 'seen' at training time*. 
 - *this may result in unseen fandoms at validation time though, for instance (A1, F1, A1, F2) at training time and (A1, F3, A1, F4) at validation 


| Dataset      | Number of pairs | Positive | Negative
| ----------- | ----------- | ---------| --------|
| PAN 2020 XL | 275565 | 147778 | 127787 |
| PAN 2020 XL train | 261786 | 140389 | 121397 |
| PAN 2020 XL val | 13779 | 7389 | 6390 |
| PAN 2020 XS | 52601 | 27834 | 24767 |
| PAN 2020 XS train | 47342 | 25051 | 22291 |
| PAN 2020 XS val | 5259 | 2783 | 2476 |
 
### Open-set setup
We make sure that authors or fandoms in the training split do no appear in the validation split.

| Dataset      | Number of pairs | Positive | Negative
| ----------- | ----------- | ---------| --------|
| PAN 2020 XL | 275565 | 147778 | 127787 |
| PAN 2020 XL train | TODO | TODO | TODO |
| PAN 2020 XL val | TODO | TODO | TODO |
| PAN 2020 XS | 52601 | 27834 | 24767 |
| PAN 2020 XS train | TODO | TODO | TODO |
| PAN 2020 XS val | TODO | TODO | TODO |



## PAN 2020 XS (Small dataset)
 Data file: ```pan20-authorship-verification-training-small.jsonl```

 Ground truth file: ```pan20-authorship-verification-training-small-truth.jsonl```

 52601 pairs, 27834 are positive (same-author) and 24767 are negative (different authors)

 We concatenate the data and ground truth files into a single file ```pan20-av-small.jsonl``` by calling the ```merge_data_and_labels()``` function.

### Train/validation splits
 We split ```pan20-av-small.jsonl``` into a 90% training set and a 10% validation set using the ```split_train_val()``` function.

 Training file: ```pan20-av-small-train.jsonl```
 Validation file: ```pan20-av-small-val.jsonl```

 We take the first 20% of the positive examples. For each k pairs of a single author, we store k/2 pairs into the training set and k/2 pairs into the validation set. This results in a validation set with 10% positive pairs whose authors are found in the training set as well. The rest of the 80% positive examples are stored in the training set.

Training size: 47342, positive: 25051, negative: 22291
Validation size: 5259, positive: 2783, negative: 2476

## PAN 2020 XL (Large dataset)
 Data file: ```pan20-authorship-verification-training-large.jsonl```

 Ground truth file: ```pan20-authorship-verification-training-large-truth.jsonl```

  275565 pairs, 147778 are positive (same-author) and 127787 are negative (different authors)

  We concatenate the data and ground truth files into a single file ```pan20-av-large.jsonl``` by calling the ```merge_data_and_labels()``` function.

  ### Train/validation splits
 We split ```pan20-av-large.jsonl``` into a 95% training set and a 5% validation set using the ```split_train_val()``` function.

 Training file: ```pan20-av-large-train.jsonl```
 Validation file: ```pan20-av-large-val.jsonl```

Training size: 261786*, positive: 140389, negative: 121397
Validation size: 13779*, positive: 7389, negative: 6390
Total: 275565
Train+val files = 275489

*Some entries have the same unique "id", so the final train/val sizes are: 261715/13774

### Storing examples in separate ```.json``` files
 Since the ```.jsonl``` files are quite large, we use the ```split_large_jsonl()``` function to store examples from ```pan20-av-large-*.jsonl``` into separate ```.json``` files.