# based on https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
# adapted by nikson

SOS_token = 0
EOS_token = 1

class BoW:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {"SOS":1,  "EOS":1}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        words = sentence if type(sentence) == list else sentence.split(' ')
        for word in words:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def seq2idx(self, s):
        words = s if type(s) == list else s.split(' ')
        idx = [self.word2index[word] for word in words]
        idx.append(EOS_token)
        return idx
    
    def seq2tensor(self, s, tfidf=False):
        indx = self.seq2idx(s)
        tensor = [0]*(self.n_words)
        for i in indx:
            tensor[i] += 1.0 if not tfidf else 1.0/self.word2count[self.index2word[i]]
        return tensor



        