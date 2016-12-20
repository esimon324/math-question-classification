from __future__ import division
import sys
import csv
import os
import re
import util

# object for determining important statistics on given dataset
class Analyzer:
    stop_words = []
    data = []
    label_set = []
    
    def __init__(self,subdir,fname):
        # read in the stopwords file
        stop_words_file = open(os.path.join('data/','stop_words'))
        self.stop_words = [line.rstrip('\n') for line in stop_words_file] #striping each \n from stop words
        self.data = util.parse_data(subdir,fname)
        
        # extract the unique labels present in the data
        self.label_set = self.get_label_set()
    
    # returns the number of unique tags that appear in the dataset
    def total_label_types(self):
        labels = []
        for tokens,tags in self.data:
            for tag in tags:
                if tag not in labels:
                    labels.append(tag)
        return len(labels)
    
    # returns the total number of tokens in the dataset
    def total_tokens(self):
        token_count = 0
        for tokens,tags in self.data:
            token_count += len(tokens)
        return token_count
        
    # returns the total number of types in the dataset
    def total_types(self):
        types = []
        for tokens,tags in self.data:
            for token in tokens:
                if token not in types:
                    types.append(token)
        return len(types)
    
    # returns the total number of tokens for a given label
    def tokens_by_label(self,label):
        count = 0
        for tokens,tags in self.data:
            if label in tags:
                count += len(tokens)
        return count
    
    # returns the total number of types for a given label
    def types_by_label(self,label):
        types = []
        for tokens,tags in self.data:
            if label in tags:
                for token in tokens:
                    if token not in types:
                        types.append(token)
        return len(types)
    
    # returns a dict of tokens to their frequencies
    def tokens2freq(self):
        freqs = {}
        for tokens,tags in self.data:
            for token in tokens:
                if token not in freqs:
                    freqs[token] = 0
                freqs[token] += 1
        return freqs
    
    # returns a dict of tokens to their frequencies within for a given label
    def tokens2freq_by_label(self,label):
        freqs = {}
        for tokens,tags in self.data:
            if label in tags:
                for token in tokens:
                    if token not in freqs:
                        freqs[token] = 0
                    freqs[token] += 1
        return freqs
        
    # returns n most frequent words for given label from dataset as a list of tuples
    def most_freq_words_by_label(self,label,n,stopwords=True):
        freq = {}
        # for each post
        for post,tag in self.data:
            if label in tag:
                # for each token in post as a tokenized list
                for token in post:
                    # apply stop word filtering
                    if not stopwords or token not in self.stop_words:
                        # increment frequency of token
                        if token not in freq:
                            freq[token] = 0
                        freq[token] += 1
                
        # express freq dict as list of tuples and sort in desc order        
        freq_n_list = freq.items()
        freq_n_list.sort(key=lambda tup: tup[1],reverse=True)
        
        # return first n items as n most frequent
        return freq_n_list[:n]
    
    # returns n most frequent words in the dataset as a list of tuples
    def most_freq_words(self,n,stopwords=True):
        freq = {}
        # for each post
        for post,tag in self.data:
            # for each token in post
            for token in post:
                # apply stop word filtering
                if not stopwords or token not in self.stop_words:
                    # increment frequency of token
                    if token not in freq:
                        freq[token] = 0
                    freq[token] += 1
                
        # express freq dict as list of tuples and sort in desc order        
        freq_n_list = freq.items()
        freq_n_list.sort(key=lambda tup: tup[1],reverse=True)
        
        # return first n items as n most frequent
        return freq_n_list[:n]
    
    # returns a list of all tags in the data set
    def get_label_set(self):
        label_set = []
        for post,tags in self.data:
            for tag in tags:
                if tag not in label_set:
                    label_set.append(tag)
        return label_set
    
    # returns a dict of tags to top n most frequency words
    def all_label_keywords(self,n):
        label2keywords = {}
        for label in self.label_set:
            label2keywords[label] = []
            top_n = self.most_freq_words_by_label(label,n)
            for word,freq in top_n:
                label2keywords[label].append(word)
            
        return label2keywords
    
    # returns the average size of the tagset per sample
    def mean_tag_set_size(self):
        sum = 0
        for post,tags in self.data:
            sum += len(tags)
        return sum / len(self.data)