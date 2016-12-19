from __future__ import division
import nltk
import csv
import re
import random
import math
import sys
import os
import util

from keyword_frequency_classifier import KeywordFrequencyClassifier
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from nltk import NaiveBayesClassifier
    
if __name__ == "__main__":
	# read in the data set
    subdir = 'data/single_tags/'
    fname = 'dataset.csv'
    data = util.parse_data(subdir,fname,single_label=True,extract_features=True)
    
    # randomize the data cases
    random.shuffle(data)
    
    # split into training and testing data
    slice = math.trunc(len(data)*(.8)) # 80% train, 20% test
    train_set = data[:slice]
    test_set = data[slice:]

    # train classification models
    print 'Training models on',len(train_set),'data samples...'
    nb = NaiveBayesClassifier.train(train_set)
    lr = SklearnClassifier(LogisticRegression()).train(train_set)
    kwfc = KeywordFrequencyClassifier()
    kwfc.train(train_set)

    # calculate and report model accuracy
    print '\nKey Word Frequency Classifier accuracy based on',len(test_set),'samples:'
    print kwfc.accuracy(test_set)
    
    print '\nNaive Bayes accuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(nb,test_set)
    
    print '\nLogistic Regression accuracy based on',len(test_set),'samples:'
    print nltk.classify.util.accuracy(lr,test_set)
    
    # extracting sample sentence from command line for classification
    sample_post = 'How many numbers less than 70 are relatively prime to it?'
    for token in sys.argv[1:]:
        sample_post = sample_post + token + ' '
    test = util.features(sample_post)
    
    # attempt to classsify sample sentence
    print '\nAttempting to Classify:\n',sample_post
    dist = nb.prob_classify(test)
    print 'Naive Bayes:',nb.classify(test)
    print 'Keyword Classifier',kwfc.predict(test)
    print 'Logistic Regression:',lr.classify(test)