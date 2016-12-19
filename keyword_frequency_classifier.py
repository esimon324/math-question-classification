from __future__ import division
import os
import re
import operator

class KeywordFrequencyClassifier:
    label_key_words = {}
    labels = []
    stop_words = []
    
    def __init__(self,n=100,pseudocount=1):
        self.pseudocount = pseudocount
        self.n = n
        self.get_stop_words()
    
    # get stop words from data file
    def get_stop_words(self):
        stop_words_file = open(os.path.join('data/','stop_words'))
        self.stop_words = [line.rstrip('\n') for line in stop_words_file]
    
    # train model to prepare for prediction
    def train(self,train_set):
        class_word_freq = {}
        # form a dict of word to frequency in training set
        for text,label in train_set:
            if label not in self.labels:
                # adding new label to label set
                self.labels.append(label)
                self.label_key_words[label] = []
                class_word_freq[label] = {}
            for word in text:
                # apply stop word filtering
                if word not in self.stop_words:
                    # increment frequency of word
                    if word not in class_word_freq[label]:
                        class_word_freq[label][word] = 0
                    class_word_freq[label][word] += 1
                    
        # choose n most frequent words from the the dict
        for label in class_word_freq:
            top_n = class_word_freq[label].items()
            top_n.sort(key=lambda tup: tup[1],reverse=True)
            top_n = top_n[:(self.n)]
            # form a dict of label to list of n most frequent words
            for word,freq in top_n:
                self.label_key_words[label].append(word)
    
    # predicts label of given sample
    def predict(self,sample):
        label_scores = {}
        for label in self.labels:
            label_scores[label] = self.pseudocount
            
        for word in sample:
            for label in self.labels:
                if word in self.label_key_words[label]:
                    label_scores[label] += 1
                    
        return max(label_scores.iteritems(), key=operator.itemgetter(1))[0]
    
    # teruns ratio of correctly classified cases out of all cases in test set
    def accuracy(self,test_data):
        num_correct = 0
        for words,label in test_data:
            if self.predict(words) == label:
                num_correct += 1
        return num_correct / len(test_data)