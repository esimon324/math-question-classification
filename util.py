from __future__ import division
import sys
import csv
import os
import re
import random

# extracts features from a given body of text
def features(text,latex=True):
    features = {}
    if latex:
        features['latex_symbol'] = 0
    for token in tokenize(text):
        # bow feature extraction
        if token not in features:
            features[token] = 0
        # word count features
        features[token] += 1
        # LaTex feature extraction
        if latex:
            if '%' in token:
                features['latex_symbol'] += 1

    return features  

# converts a string to a list of lowercased tokens
def tokenize(text):
    text = text.lower()
    text = re.sub(r'<[\w/]*>','',text)
    tokens = text.split()
    return tokens

# converts label tag strings into a list of labels
def extract_labels(labels):
    return re.sub(r'<|>',' ',labels).split()

# parses csv into shuffled dataset with features already extracted
def parse_data(subdir,fname,single_label=False,extract_features=False):
    # read in the data set
    csvfile = open(os.path.join(subdir, fname))
    reader = csv.reader(csvfile,delimiter=',')
    raw_data = list(reader)
    
    # collect features and label to form dataset
    dataset = []
    for post,tags in raw_data:
        y = extract_labels(tags)
        if extract_features:
            x = features(post)
            if single_label:
                y = y[0]
        else:
            x = tokenize(post)
        dataset.append((x,y))
    
    # randomize the data cases
    random.shuffle(dataset)
    
    return dataset
    
# number of bit flips to get from prediction to gold standard (total error)    
def hamming_error(gold,pred):
    count = 0
    for i in range(len(gold)):
        if gold[i] != pred[i]:
            count += 1
    return count

# number of tags missed by OVR classifier for given gold standard and prediction
def recall_error(gold,pred):
    count = 0
    for i in range(len(gold)):
        if gold[i] == 1 and pred[i] == 0:
            count += 1
    return count

# number of tags wrongly predicted by OVR classifier given gold standard and prediction
def precision_error(gold,pred):
    count = 0
    for i in range(len(gold)):
        if gold[i] == 0 and pred[i] == 1:
            count += 1
    return count