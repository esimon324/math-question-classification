import nltk
import csv
import re
import random
import math
import sys
import os

from key_word_frequency_classifier import KeyWordFrequencyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from nltk.classify.scikitlearn import SklearnClassifier
from nltk import NaiveBayesClassifier
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
    for token in post:
        if token not in features:
            features[token] = 0
        features[token] += 1
    return features
### Utility Functions ###
    
if __name__ == "__main__":
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
        train_set.append((features(post),tag))
        kwfc_train_set.append((post,tag))
        i += 1

    # collect features and label from each test case
    test_set = []
    kwfc_test_set = []
    for datacase in test_data:
        post,tag = datacase
        post = tokenize(post)
        test_set.append((features(post),tag))
        kwfc_test_set.append((post,tag))

    # train a simple Naive Bayes model
    print 'Training Naive Bayes model on',len(train_set),'data samples...'
    nb = NaiveBayesClassifier.train(train_set)
    lr = SklearnClassifier(LogisticRegression()).train(train_set)
    svc = SklearnClassifier(LinearSVC()).train(train_set)
    kwfc = KeyWordFrequencyClassifier('stop_words')
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
    print '\nNaive Bayes accuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(nb,test_set)
    
    print '\nLogistic Regression ccuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(lr,test_set)
    
    print '\nLinear Support Vector Machine accuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(svc,test_set)
    
    print '\nKey Word Frequency Classifier accuracy based on',len(kwfc_test_set),'samples:'
    print kwfc.accuracy(kwfc_test_set)
    