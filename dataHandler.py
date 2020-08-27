# based on https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
# adapted by nikson

import csv
import unicodedata
import re
from boW import BoW
from nltk.stem.snowball import SnowballStemmer

class DataHandler:
    def __init__(self):
        self.askDictionary = BoW('ask')
        self.ansDictionary= BoW('answers')
        self._stopWords = self.readStopWords()
        self._porter = SnowballStemmer("portuguese")
    
    def readStopWords(self):
        with open('stopWords', 'rt', encoding='utf-8') as f:
            words = f.read().split('\n')
        return words[:-1]

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalize(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" ", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def preProcess(self, s):
        s = self.normalize(s)
        words = s.strip().split(' ')
        #remove stopwords
        words = [w for w in words if not w in self._stopWords]
        #stem 
        stemmed = [self._porter.stem(word) for word in words]
        return stemmed

    def readData(self):
        with open('dataset.csv', 'rt', encoding='utf-8') as f:
            data = list(csv.reader(f))
        return data

    def readDataPreproc(self, pre=False):
        data = self.readData()

        data = [[self.preProcess(d[0]), \
                 self.normalize(d[1]), d[2]] for d in data] \
                if not pre else \
                [[self.preProcess(d[0]), \
                self.preProcess(d[1]), d[2]] for d in data]                    
        return data

    def createDictionary(self):
        data = self.readData()
        for (ask,ans,emo) in data:
            self.ansDictionary.addSentence(self.normalize(ans + " " + ask))
            self.askDictionary.addSentence(self.preProcess(ans + " " + ask))
        print("Ask dictionary length: {}".format(self.askDictionary.n_words))
        print("Answers dictionary length: {}".format(self.ansDictionary.n_words))

if __name__== '__main__':
    d = DataHandler()
    d.createDictionary()
