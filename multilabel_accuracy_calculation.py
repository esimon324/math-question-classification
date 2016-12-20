from __future__ import division
import util
import random
import sys
import math
from one_vs_rest import OneVsRestClassifier

# performs a number of command line specified trials to find experimental error of OVR
# using all six evaluation metrics. These six metrics are averaged over all trials to 
# reduce the variance of reported error.
def main():  
    # read in the data set
    subdir = 'data/original_tags/'
    fname = 'dataset.csv'
    data = util.parse_data(subdir,fname,extract_features=True)
    
    num_trials = int(sys.argv[1])
    
    trial_results = {}
    trial_results['sum_total_hamming'] = 0
    trial_results['sum_total_precision'] = 0
    trial_results['sum_total_recall'] = 0
    
    trial_results['sum_mean_hamming'] = 0
    trial_results['sum_mean_precision'] = 0
    trial_results['sum_mean_recall'] = 0
    
    print 'Trial ',
    # run the trials
    for i in range(num_trials):
        print (i+1),
        # randomize the data cases
        random.shuffle(data)
        
        # split into training and testing data
        slice = math.trunc(len(data)*(.8)) # 80% train, 20% test
        train_set = data[:slice]
        test_set = data[slice:]
        
        # train a new classifier
        ovr = OneVsRestClassifier()
        ovr.fit(train_set)
        
        # determine total error for each metric
        total_hamming_error = ovr.total_hamming_error(test_set)
        total_precision_error = ovr.total_precision_error(test_set)
        total_recall_error = ovr.total_recall_error(test_set)
        n = len(test_set)
        
        # update relevant error entries in the dictionary
        trial_results['sum_total_hamming'] += total_hamming_error
        trial_results['sum_total_precision'] += total_precision_error
        trial_results['sum_total_recall'] += total_recall_error
        
        trial_results['sum_mean_hamming'] += ( total_hamming_error / n )
        trial_results['sum_mean_precision'] += ( total_precision_error / n )
        trial_results['sum_mean_recall'] += ( total_recall_error / n )
    
    # print the results
    print '\n---'
    print 'Number of trials:',num_trials
    for metric,value in trial_results.items():
        print metric, value
        print 'Average:',value / num_trials
        print
    
if __name__ == "__main__": 
    main()