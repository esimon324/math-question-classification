import nltk
import sklearn
import csv
import re
import random
import math

### Utility Functions ###
def extract_tags(tags):
    tags = tags.replace('<','')
    return tags.split('>')[:-1]

def tokenize(text):
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
    
csvfile = open('dataset.csv', 'rb')
reader = csv.reader(csvfile,delimiter=',')
data = list(reader)
random.shuffle(data)
slice = math.trunc(len(data)*(.8)) # 80% train, 20% test
train_data = data[:slice]
test_data = data[slice:]

train_set = []
i = 0
for datacase in train_data:
    print i
    post,tag = datacase
    post = tokenize(post)
    train_set.append((features(post),tag))
    i += 1
    
test_set = []
for datacase in test_data:
    post,tag = datacase
    post = tokenize(post)
    test_set.append((features(post),tag))

# print "Test set sample:\n",test_set[:10]
nb = nltk.NaiveBayesClassifier.train(train_set)
# sample_post = tokenize('What is the moment generating function for the binomial distribution?')
sample_post = 'how do I calculate the area of an equilateral triangle?'
sample_tokens = tokenize(sample_post)
test = features(sample_tokens)
print 'Attempting to Classify:\n',sample_post
print nb.classify(test)

print '\nModel Accuracy:',
print nltk.classify.util.accuracy(nb,test_set)