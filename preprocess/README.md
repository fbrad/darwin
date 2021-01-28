

# PAN 2020 Dataset statistics

## Small dataset
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

## Large dataset
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

*Some entries have the same unique "id", so the final train/val sizes are: 261715/13774

### Storing examples in separate ```.json``` files
 Since the ```.jsonl``` files are quite large, we use the ```split_large_jsonl()``` function to store examples from ```pan20-av-large-*.jsonl``` into separate ```.json``` files.