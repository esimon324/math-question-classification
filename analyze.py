import sys
import csv
import os
import question_classifier as qc

# returns n most frequent words in the csv data set as a list of tuples
def word_freq_by_class(n,data,stopwords=True):
    freq = {}
    # for each post
    for post,label in data:
        # for each token in post as a tokenized list
        for token in qc.tokenize(post):
            # apply stop word filtering
            if not stopwords or token not in STOP_WORDS:
                # increment frequency of token
                if token not in freq:
                    freq[token] = 0
                freq[token] += 1
            
    # express freq dict as list of tuples and sort in desc order        
    freq_n_list = freq.items()
    freq_n_list.sort(key=lambda tup: tup[1],reverse=True)
    
    # return first n items as n most frequent
    return freq_n_list[:n]
    
def main():    
    # read in the stopwords file
    stop_words_file = open(os.path.join('data/','stop_words'))
    global STOP_WORDS
    STOP_WORDS = [line.rstrip('\n') for line in stop_words_file] #striping each \n from stop words
    
    # cmd line specified number of top words
    n = int(sys.argv[2])
    
    # read in the data set csv denoted in the command line
    data_fname = sys.argv[1]
    csvfile = open(os.path.join('data/single_tags/', data_fname))
    reader = csv.reader(csvfile,delimiter=',')
    data = list(reader)
    
    # calculate the top n words
    top_n = word_freq_by_class(n,data)
    
    # write results to file
    fname = (data_fname.split('.'))[0]+'_frequencies_top_'+str(n)
    outfile = open(fname,'w')
    for word,freq in top_n:
        outfile.write(word+'  '+str(freq)+'\n')
    
    print top_n
    
if __name__ == "__main__":
    main()