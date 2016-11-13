import sys
import csv
import os
import re

class Analyzer:
    label_key_words = {}
    stop_words = []
    data = []
    
    def __init__(self):
        # read in the stopwords file
        stop_words_file = open(os.path.join('data/','stop_words'))
        self.stop_words = [line.rstrip('\n') for line in stop_words_file] #striping each \n from stop words
        
        # read in the data set
        csvfile = open(os.path.join('data/single_tags/','dataset.csv'))
        reader = csv.reader(csvfile,delimiter=',')
        self.data = list(reader)
    
    def tokenize(self,text):
        text = text.lower()
        text = re.sub(r'<[\w/]*>','',text)
        tokens = text.split()
        return tokens
        
    # returns n most frequent words for given label from dataset as a list of tuples
    def most_freq_words_by_label(self,label,n,stopwords=True):
        freq = {}
        # for each post
        for post,tag in self.data:
            if tag == label:
                # for each token in post as a tokenized list
                for token in self.tokenize(post):
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
    
    def most_freq_words(self,n,stopwords=True):
        freq = {}
        # for each post
        for post,tag in self.data:
            # for each token in post as a tokenized list
            for token in self.tokenize(post):
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
        
def main():      
    # cmd line specified data file
    data_fname = sys.argv[1]

    # cmd line specified number of top words
    n = int(sys.argv[2])

    # cmd line speficied label
    label = sys.argv[3]  
    
    # cmd line specified desire to write results to file
    write_to_file = sys.argv[4]
    
        
    # creating analyzer object
    a = Analyzer()
    
    # if label provided, calculate n most frequent words per label
    top_n = []
    if label != "all":
        top_n = a.most_freq_words_by_label(label,n)
    else:
        top_n = a.most_freq_words(n)
    
    # write results to file
    if write_to_file:
        fname = (data_fname.split('.'))[0]+'_frequencies_top_'+str(n)
        outfile = open(fname,'w')
        for word,freq in top_n:
            outfile.write(word+'  '+str(freq)+'\n')
    
    print top_n
    
if __name__ == "__main__":
    main()