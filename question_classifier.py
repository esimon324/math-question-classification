import nltk
import sklearn
import csv
import re
import random
import math
import sys

### Utility Functions ###
def extract_tags(tags):
    tags = tags.replace('<','')
    return tags.split('>')[:-1]

def tokenize(text):
    text.lower()
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
    csvfile = open('single_tag_dataset.csv', 'rb')
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
    i = 0
    for datacase in train_data:
        # print i
        post,tag = datacase
        post = tokenize(post)
        train_set.append((features(post),tag))
        i += 1

    # collect features and label from each test case
    test_set = []
    for datacase in test_data:
        post,tag = datacase
        post = tokenize(post)
        test_set.append((features(post),tag))

    # train a simple Naive Bayes model
    print 'Training Naive Bayes model on',len(train_set),' data samples...'
    nb = nltk.NaiveBayesClassifier.train(train_set)
	
    # extracting sample sentence from command line for classification
    sample_post = ''
    for token in sys.argv[1:]:
        sample_post = sample_post + token + ' '
    sample_tokens = tokenize(sample_post)
    test = features(sample_tokens)
    
    # attempt to classsify sample sentence
    print 'Attempting to Classify:\n',sample_post
    print nb.classify(test)

    # calculate and report model accuracy
    print '\nModel Accuracy:',
    print nltk.classify.util.accuracy(nb,test_set)
	
    