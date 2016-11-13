import nltk
import csv
import re
import random
import math
import sys
import os

from keyword_frequency_classifier import KeywordFrequencyClassifier
from analyze import Analyzer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from nltk.classify.scikitlearn import SklearnClassifier
from nltk import NaiveBayesClassifier

### Global Variables ###
LABELS = ['calculus','category_theory','combinatorics','geometry','graph_theory','linear_algebra','logic','number_theory','probability']
### Global Variables ###

### Utility Functions ###
def extract_tags(tags):
    tags = tags.replace('<','')
    return tags.split('>')[:-1]

def tokenize(text):
    text = text.lower()
    text = re.sub(r'<[\w/]*>','',text)
    tokens = text.split()
    return tokens

def features(post):
    features = {}
    # word count features
    for token in post:
        if token not in features:
            features[token] = 0
        features[token] += 1
    return features

def features2(post):
    features = {'latex_symbol':0}
    for token in post:
        # word count features
        if token not in features:
            features[token] = 0
        # LaTex symbol count features
        features[token] += 1
        if '%' in token:
            features['latex_symbol'] += 1
    return features

def features3(post):
    features = {'latex_symbol':0}
    for token in post:
        # word count features
        if token not in features:
            features[token] = 0
        features[token] += 1
        
        # LaTex symbol count feature
        if '%' in token:
            features['latex_symbol'] += 1
        
        # keyword count feature
        for label in LABELS:
            if token in LABEL_KEYWORDS[label]:
                key = label+'_keyword_count'
                if key not in features:
                    features[key] = 0
                features[key] += 1
                
    return features
### Utility Functions ###
    
if __name__ == "__main__":
    # calculate top 100 most frequent words per label for use in feature space
    a = Analyzer()
    global LABEL_KEYWORDS
    LABEL_KEYWORDS = {}
    for label in LABELS:
        LABEL_KEYWORDS[label] = []
        top_n = a.most_freq_words_by_label(label,100)
        for word,freq in top_n:
            LABEL_KEYWORDS[label].append(word)

	# read in the data set
    subdir = 'data/single_tags/'
    fname = 'dataset.csv'
    csvfile = open(os.path.join(subdir, fname))
    reader = csv.reader(csvfile,delimiter=',')
    data = list(reader)
    
    # randomize the data cases
    random.shuffle(data)
    
    # split into training and testing data
    slice = math.trunc(len(data)*(.8)) # 80% train, 20% test
    train_data = data[:slice]
    test_data = data[slice:]
    
    # collect features and label from each training case
    train_set = []
    kwfc_train_set = []
    i = 0
    for datacase in train_data:
        # print i
        post,tag = datacase
        post = tokenize(post)
        train_set.append((features2(post),tag))
        kwfc_train_set.append((post,tag))
        i += 1

    # collect features and label from each test case
    test_set = []
    kwfc_test_set = []
    for datacase in test_data:
        post,tag = datacase
        post = tokenize(post)
        test_set.append((features2(post),tag))
        kwfc_test_set.append((post,tag))

    # train a simple Naive Bayes model
    print 'Training models on',len(train_set),'data samples...'
    nb = NaiveBayesClassifier.train(train_set)
    lr = SklearnClassifier(LogisticRegression()).train(train_set)
    svc = SklearnClassifier(LinearSVC()).train(train_set)
    kwfc = KeywordFrequencyClassifier('stop_words')
    kwfc.train(kwfc_train_set)
    
    # extracting sample sentence from command line for classification
    sample_post = ''
    for token in sys.argv[1:]:
        sample_post = sample_post + token + ' '
    sample_tokens = tokenize(sample_post)
    test = features(sample_tokens)
    
    # attempt to classsify sample sentence
    print '\nAttempting to Classify:\n',sample_post
    dist = nb.prob_classify(test)
    print nb.classify(test)
    
    # print confidence distribution over classes
    print '\nClassifier confidence for above post:'
    for sample in dist.samples():
        print sample,' ',dist.prob(sample)

    # calculate and report model accuracy
    print '\nKey Word Frequency Classifier accuracy based on',len(kwfc_test_set),'samples:'
    print kwfc.accuracy(kwfc_test_set)
    
    print '\nNaive Bayes accuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(nb,test_set)
    
    print '\nLogistic Regression ccuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(lr,test_set)
    