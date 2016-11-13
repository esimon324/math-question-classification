import nltk
import csv
import re
import random
import math
import sys
import os

from key_word_classifier import KeyWordClassifier

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
    kwc_train_set = []
    i = 0
    for datacase in train_data:
        # print i
        post,tag = datacase
        post = tokenize(post)
        train_set.append((features(post),tag))
        kwc_train_set.append((post,tag))
        i += 1

    # collect features and label from each test case
    test_set = []
    kwc_test_set = []
    for datacase in test_data:
        post,tag = datacase
        post = tokenize(post)
        test_set.append((features(post),tag))
        kwc_test_set.append((post,tag))

    # train a simple Naive Bayes model
    print 'Training Naive Bayes model on',len(train_set),'data samples...'
    nb = NaiveBayesClassifier.train(train_set)
    lr = SklearnClassifier(LogisticRegression()).train(train_set)
    svc = SklearnClassifier(LinearSVC()).train(train_set)
    kwc = KeyWordClassifier('stop_words')
    kwc.train(kwc_train_set)
    
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
    print '\nNaive Bayes Accuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(nb,test_set)
    
    print '\nLogistic Regression Accuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(lr,test_set)
    
    print '\nLinear Support Vector Machine Accuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(svc,test_set)
    
    print '\nKey Word Classifier Accuracy based on',len(kwc_test_set),'samples:'
    print kwc.accuracy(kwc_test_set)
    