import sys
import csv
import os
import re

def features(text,latex=True):
    features = {}
    if latex:
        features['latex_symbol'] = 0
    for token in tokenize(text):
        if token not in features:
            features[token] = 0
        # word count features
        features[token] += 1
        if latex:
            if '%' in token:
                features['latex_symbol'] += 1

    return features  
    
def tokenize(text):
    text = text.lower()
    text = re.sub(r'<[\w/]*>','',text)
    tokens = text.split()
    return tokens
    
def extract_labels(self,labels):
    return re.sub(r'<|>',' ',labels).split()
    
def hamming_error(gold,pred):
    count = 0
    for i in range(len(gold)):
        if gold[i] != pred[i]:
            count += 1
    return count