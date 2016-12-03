import sys
import csv
import os
import re

class Analyzer:
    label_key_words = {}
    stop_words = []
    data = []
    label_set = []
    
    def __init__(self,data=None):
        # read in the stopwords file
        stop_words_file = open(os.path.join('data/','stop_words'))
        self.stop_words = [line.rstrip('\n') for line in stop_words_file] #striping each \n from stop words
            
        if data == None:
            # read in the data set
            csvfile = open(os.path.join('data/original_tags/','dataset.csv'))
            reader = csv.reader(csvfile,delimiter=',')
            self.data = list(reader)
        else:
            self.data = data
        
        self.label_set = self.get_label_set()
    
    def features_wc(self,text):
        text = self.tokenize(text)
        features = {}
        # word count features
        for token in text:
            if token not in features:
                features[token] = 0
            features[token] += 1
        return features  
        
    def tokenize(self,text):
        text = text.lower()
        text = re.sub(r'<[\w/]*>','',text)
        tokens = text.split()
        return tokens
    
    def extract_labels(self,labels):
        return re.sub(r'<|>',' ',labels).split()
        
    # returns n most frequent words for given label from dataset as a list of tuples
    def most_freq_words_by_label(self,label,n,stopwords=True):
        freq = {}
        # for each post
        for post,tag in self.data:
            if label in tag:
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
    
    def get_label_set(self):
        set = []
        for post,tags in self.data:
            for tag in self.extract_labels(tags):
                if tag not in set and tag != '':
                    set.append(tag)
        return set
    
    # returns a dict of tags to top n most frequency words
    def all_label_keywords(self,n):
        label2keywords = {}
        for label in self.label_set:
            label2keywords[label] = []
            top_n = self.most_freq_words_by_label(label,n)
            for word,freq in top_n:
                label2keywords[label].append(word)
            
        return label2keywords
    
    # returns the average size of the tagset for each sample in the data
    def mean_tag_set_size(self):
        sum = 0
        for post,tags in self.data:
            sum += len(self.extract_labels(tags))
        return sum / len(self.data)
        
def main():      
    # # cmd line specified data file
    # data_fname = sys.argv[1]

    # # cmd line specified number of top words
    # n = int(sys.argv[2])

    # # cmd line speficied label
    # label = sys.argv[3]  
    
    # # cmd line specified desire to write results to file
    # write_to_file = sys.argv[4]
    
        
    # # creating analyzer object
    # a = Analyzer()
    
    # # if label provided, calculate n most frequent words per label
    # top_n = []
    # if label != "all":
        # top_n = a.most_freq_words_by_label(label,n)
    # else:
        # top_n = a.most_freq_words(n)
    
    # # write results to file
    # if write_to_file:
        # fname = (data_fname.split('.'))[0]+'_frequencies_top_'+str(n)
        # outfile = open(fname,'w')
        # for word,freq in top_n:
            # outfile.write(word+'  '+str(freq)+'\n')
    
    a = Analyzer()
    
    # print top_n
    
if __name__ == "__main__":
    main()