from __future__ import division
import util
import os
import csv
import random
import math
from analyze import Analyzer
from one_vs_rest import OneVsRestClassifier

# analysis of multi-label dataset and OVR classifier
def main():  
    # read in the data set
    subdir = 'data/original_tags/'
    fname = 'dataset.csv'
    data = util.parse_data(subdir,fname,extract_features=True)
    
    # randomize the data cases
    random.shuffle(data)
    
    # split into training and testing data
    slice = math.trunc(len(data)*(.8)) # 80% train, 20% test
    train_data = data[:slice]
    test_data = data[slice:]
    
    # instantiating classifier
    ovr = OneVsRestClassifier()
    a = Analyzer(subdir,fname)
    
    # printing dataset statistics
    print 'Dataset Statistics\n---'
    print 'Total Tokens:',a.total_tokens()
    print 'Total Types:',a.total_types()
    print 'Total Label Types:',a.total_label_types()
    print 'Average number of tags per sample:',a.mean_tag_set_size()
    print 
    
    # printing OVR training specific statistics
    print 'Training Statistics\n---'
    ovr.fit(train_data,threshold=200,print_stats=True)   
    
    total_hamming_error = ovr.total_hamming_error(test_data)
    total_recall_error = ovr.total_recall_error(test_data)
    total_precision_error =ovr.total_precision_error(test_data)
    test_size = len(test_data)
    
    print
    print 'Model Accuracy\n---'
    print 'Total Hamming Error:',total_hamming_error
    print 'Mean Hamming Error:',total_hamming_error / test_size
    print 'Total Recall Error:',total_recall_error
    print 'Mean Recall Error:',total_recall_error / test_size
    print 'Total Precision Error:',total_precision_error
    print 'Mean Precision Error:',total_precision_error / test_size   
    print
    
    # An example
    sample_str = 'How many numbers less than 70 are relatively prime to it?'
    sample = util.features(sample_str)
    gold_y = ovr.transform(['combinatorics','number-theory'])
    
    print 'An Example\n---'
    print sample_str
    pred_y = ovr.predict(sample)
    print 'Prediction:',ovr.inverse_transform(pred_y)
    print 'Actual:',ovr.inverse_transform(gold_y)
    print 'Hamming Error:',util.hamming_error(gold_y,pred_y)
    print 'Recall Error:',util.recall_error(gold_y,pred_y)
    print 'Precision Error:',util.precision_error(gold_y,pred_y)

if __name__ == "__main__": 
    main()