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

class OneVsRestClassifier:
    classifiers = {}
    label_set = []
    
    def __init__(self):
        pass
    
    def fit(self,train_set,one2one=True,threshold=200):
        a = Analyzer(train_set)
        num_removed = 0
        self.label_set = a.get_label_set()
        if one2one:
            for label in self.label_set:
                cur_train = []
                for x,y in train_set:
                    if label in y:
                        cur_train.append((x,1))
                num_pos = len(cur_train)
                if num_pos < threshold:
                    self.classifiers[label] = None
                    num_removed += 1
                else:
                    num_neg = 0
                    for x,y in train_set:
                        if label not in y and num_neg < num_pos:
                            cur_train.append((x,0))
                            num_neg += 1
                    random.shuffle(cur_train)
                    self.classifiers[label] = SklearnClassifier(LogisticRegression()).train(cur_train)
            print num_removed
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
        
    def predict_dict(self,sample):
        pred = {}
        for label in self.label_set:
            pred[label] = self.classifiers[label].classify(sample)
        return pred
    
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
    
    def inverse_transform(self,bin_list):
        tag_list = []
        for i in range(len(bin_list)):
            if bin_list[i] == 1:
                tag_list.append(self.label_set[i])
        return tag_list
    
    def mean_hamming_error(self,test_set):
        sum_error = 0
        for x,y in test_set:
            gold = self.transform(y)
            pred = self.predict(x)
            # print self.inverse_transform(pred)
            sum_error += util.hamming_error(gold,pred)
        return sum_error / len(test_set)    
        
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
    print len(Analyzer(data).get_label_set())
    
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
    
    ovr.fit(train_set)
    
    sample_str = 'What how many distinct even numbers sum to primes?'
    sample = a.features_wc(sample_str)
    
    print 'Predicting:',sample_str
    print ovr.inverse_transform(ovr.predict(sample))
    print 'Error',ovr.mean_hamming_error(test_set)
    # print 'Average tag length:',a.mean_tag_set_size()
    # classif = OneVsRestClassifier()
    # expr = []
    # for i in range(20,41):
        # n = i*10
        # classif.fit(train_set,threshold=n)
        # error = classif.mean_hamming_error(test_set)
        # print n,error
        # expr.append(error)
    # for error in expr:
        # print error
    # print ovr.accuracy(test_set)
    # print ovr.classifiers['probability'].classify(sample)
    # lr = SklearnClassifier(LogisticRegression()).train(train_set)
    # print lr.classify(sample)