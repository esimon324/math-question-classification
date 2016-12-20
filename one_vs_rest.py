from __future__ import division
import os
import re
import operator
import copy
import csv
import random
import math

import util

from analyze import Analyzer
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.util import accuracy

# multi-label classifier that manifests as a collection binary classifier for each label
class OneVsRestClassifier:
    classifiers = {}
    label_set = []
    removed_labels = []
    
    def __init__(self):
        pass
    
    # trains the one vs rest model on the given training set
    def fit(self,train_set,one2one=True,threshold=200,print_stats=False):
        self.label_set = self.get_label_set(train_set)
        if one2one:
            for label in self.label_set:
                cur_train = []
                for x,y in train_set:
                    if label in y:
                        cur_train.append((x,1))
                num_pos = len(cur_train)
                if num_pos < threshold:
                    self.classifiers[label] = None
                    self.removed_labels.append(label)
                else:
                    num_neg = 0
                    for x,y in train_set:
                        if label not in y and num_neg < num_pos:
                            cur_train.append((x,0))
                            num_neg += 1
                    random.shuffle(cur_train)
                    self.classifiers[label] = SklearnClassifier(LogisticRegression()).train(cur_train)
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
        if(print_stats):
            print 'Minimum number of label occurrences required for consideration:',threshold
            print 'Total number of labels in training set:',len(self.label_set)
            print 'Number of Labels Considered:',len(self.label_set) - len(self.removed_labels)
            print 'Number of Labels Disregarded:',len(self.removed_labels)

    # given a question sample, predicts the tagset returned as a dict
    def predict_dict(self,sample):
        pred = {}
        for label in self.label_set:
            pred[label] = self.classifiers[label].classify(sample)
        return pred
    
    # given a question sample, predicts the tagset returned as a binary list
    def predict(self,sample):
        pred = []
        for label in self.label_set:
            if self.classifiers[label] == None:
                pred.append(0)
            else:
                pred.append(self.classifiers[label].classify(sample))
        return pred
    
    # converts a set of labels into a binary list
    def transform(self,tag_set):
        bin_list = [0]*len(self.label_set)
        for tag in tag_set:
            for i in range(len(self.label_set)):
                if tag == self.label_set[i]:
                    bin_list[i] = 1
        return bin_list
    
    # converts a binary list into a set of labels
    def inverse_transform(self,bin_list):
        tag_list = []
        for i in range(len(bin_list)):
            if bin_list[i] == 1:
                tag_list.append(self.label_set[i])
        return tag_list
        
    def get_label_set(self,train_set):
        label_set = []
        for feat,tags in train_set:
            for tag in tags:
                if tag not in label_set:
                    label_set.append(tag)
        return label_set
        
    def total_hamming_error(self,test_set):
        sum_error = 0
        for x,y in test_set:
            gold = self.transform(y)
            pred = self.predict(x)
            sum_error += util.hamming_error(gold,pred)
        return sum_error
    
    def mean_hamming_error(self,test_set):
        return self.total_hamming_error(test_set) / len(test_set)
        
    def total_recall_error(self,test_set):
        sum_error = 0
        for x,y in test_set:
            gold = self.transform(y)
            pred = self.predict(x)
            sum_error += util.recall_error(gold,pred)
        return sum_error
    
    def mean_recall_error(self,test_set):
        return self.total_recall_error / len(test_set)
        
    def total_precision_error(self,test_set):
        sum_error = 0
        for x,y in test_set:
            gold = self.transform(y)
            pred = self.predict(x)
            sum_error += util.precision_error(gold,pred)
        return sum_error
    
    def mean_precision_error(self,test_set):
        return self.total_precision_error / len(test_set)