from __future__ import division
import os
import re
import operator
import copy
import csv
import random
import math

from analyze import Analyzer
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.util import accuracy

class OneVsRestClassifier:
    classifiers = {}
    label_set = []
    
    def __init__(self):
        pass
    
    def fit(self,train_set,one2one=True):
        a = Analyzer(train_set)
        self.label_set = a.get_label_set()
        if one2one:
            for label in self.label_set:
                cur_train = []
                for x,y in train_set:
                    if label in y:
                        cur_train.append((x,1))
                num_pos = len(cur_train)
                num_neg = 0
                for x,y in train_set:
                    if label not in y and num_neg < num_pos:
                        cur_train.append((x,0))
                        num_neg += 1
                random.shuffle(cur_train)
                self.classifiers[label] = SklearnClassifier(LogisticRegression()).train(cur_train)
                print num_pos,len(cur_train)
        else:
            for label in self.label_set:
                cur_train = []
                num = 0
                for x,y in train_set:
                    if y == label:
                        cur_train.append((x,1))
                        num += 1
                    else:
                        cur_train.append((x,0))
                print label,num
                self.classifiers[label] = SklearnClassifier(LogisticRegression()).train(cur_train)
        
    def predict(self,sample):
        pred = {}
        for label in self.label_set:
            pred[label] = self.classifiers[label].classify(sample)
        return pred
        
    def accuracy(self,test_set):
        acc = {}
        for label in self.label_set:
            acc[label] = accuracy(self.classifiers[label],test_set)
        return acc
        
if __name__ == "__main__":
    # instantiating classifier and Analyzer
    ovr = OneVsRestClassifier()
    a = Analyzer()
    
    # read in the data set
    subdir = 'data/original_tags/'
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
    for post,tag in train_data:
        post = a.features_wc(post)
        train_set.append((post,tag))
        
    # collect features and label from each test case
    test_set = []
    for post,tag in test_data:
        post = a.features_wc(post)
        test_set.append((post,tag))
    
    print len(a.label_set)
    ovr.fit(train_set)
    
    sample_str = 'What is the probability of rolling a 3?'
    sample = a.features_wc(sample_str)
    
    print 'Predicting:',sample_str
    print ovr.predict(sample)
    print ovr.accuracy(test_set)
    # print ovr.classifiers['probability'].classify(sample)
    # lr = SklearnClassifier(LogisticRegression()).train(train_set)
    # print lr.classify(sample)