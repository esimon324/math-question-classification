import os
import sys
import csv
import pickle

from analyze import Analyzer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from nltk.classify.scikitlearn import SklearnClassifier

### UTILITY FUNCTIONS ###
def tokenize(text):
    text = text.lower()
    text = re.sub(r'<[\w/]*>','',text)
    tokens = text.split()
    return tokens
    
def features(post,key_words):
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
        for label in key_words:
            if token in key_words[label]:
                key = label+'_keyword_count'
                if key not in features:
                    features[key] = 0
                features[key] += 1
                
    return features
    
# returns a binary list representing the label set 
def labels2binary(label_set,labels):
    bin_list = [0]*len(label_set)
    for i in range(len(label_set)):
        cur_label = label_set[i]
        if cur_label in labels:
            bin_list[i] = 1
            
    return bin_list    
### UTILITY FUNCTIONS ###

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'reload':
        # read in the data set
        subdir = 'data/original_tags/'
        fname = 'dataset.csv'
        csvfile = open(os.path.join(subdir, fname))
        reader = csv.reader(csvfile,delimiter=',')
        data = list(reader)
        
        a = Analyzer(data)
        
        keywords = a.all_label_keywords(10)
        
        print 'starting loop'
        
        for i in range(len(data)):
            data[i][0] = features(data[i][0],keywords)
            data[i][1] = labels2binary(a.label_set,a.extract_labels(data[i][1]))
            print i
            
        print 'Done.'     
        with open('dataset.pickle', 'wb') as handle:
            pickle.dump(data, handle)
    else:
        print 'Reading in the data...'
        data = None
        with open('dataset.pickle', 'rb') as handle:
            data = pickle.load(handle)
        
        X = []
        Y = []
        for case,target in data:
            X.append(case)
            Y.append(target)
        
        print 'Training OneVsRestClassifier...'
        ovr = OneVsRestClassifier(LogisticRegression())
        ovr.fit(X,Y)
        
        test_str = 'What is the probability of rolling a die and getting a 3?'
        
        print 'Predicting on:',test_str
        print ovr.predict(test_set)
    # # randomize the data cases
    # random.shuffle(data)
    
    # # split into training and testing data
    # slice = math.trunc(len(data)*(.8)) # 80% train, 20% test
    # train_data = data[:slice]
    # test_data = data[slice:]
    
    # # collect features and label from each training case
    # train_set = []
    # i = 0
    # for datacase in train_data:
        # # print i
        # post,tag = datacase
        # post = tokenize(post)
        # train_set.append((features(post),tag))
        # i += 1

    # # collect features and label from each test case
    # test_set = []
    # for datacase in test_data:
        # post,tag = datacase
        # post = tokenize(post)
        # test_set.append((features2(post),tag))
        # kwfc_test_set.append((post,tag))