import os
from io import open
import torch

UNK_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'

class Dictionary(object):
    def __init__(self, minimum_count = 5):
        
        self.word2idx = {}
        self.idx2word = []
        self.word2count = {}
        self.minimum_count = minimum_count;
        self.n_words = 0;
        
        self.add_word(UNK_TOKEN, skip_mincount = True);
        self.add_word(EOS_TOKEN, skip_mincount = True);


    def add_word(self, word, skip_mincount = False):
        
        if skip_mincount:
            self.word2idx[word] = self.n_words;
            self.idx2word.append(word)
            self.n_words += 1;
        else:
            if word not in self.word2count.keys():
                self.word2count[word] = 1
            else:
                self.word2count[word] += 1
                
            if self.word2count[word] >= self.minimum_count:
                if word not in self.idx2word:
                    self.word2idx[word] = self.n_words;
                    self.idx2word.append(word)
                    self.n_words +=1 ;

    def get_index(self, word):
        if word in self.idx2word:
            return self.word2idx[word]
        else:
            return self.word2idx[UNK_TOKEN]

    def __len__(self):
        return self.n_words


class Corpus(object):
    def __init__(self, path_train, path_valid):
        self.dictionary = Dictionary()
        print('Adding words from train to dictionary..');
        self.tokenize_and_add_words(path_train)
        print('tokenizing train..');
        self.train = self.tokenize(path_train)
        print('tokenizing valid..');
        self.valid = self.tokenize(path_valid)


    
    def tokenize_and_add_words(self, path):

        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)


    def get_max_n_token(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
              
        return tokens


    def tokenize(self, path):

        tokens = self.get_max_n_token(path);
        
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.get_index(word)
                    token += 1

        return ids
    
    def tokenize_sentence(self, sentence):
       
        # Tokenize string
        token = 0
        words = sentence.split() + ['<eos>']
        ids = torch.LongTensor(len(words))
        for word in words:
            if word in list(self.dictionary.word2idx.keys()):
                ids[token] = self.dictionary.word2idx[word]
            else:
                ids[token] = self.dictionary.word2idx[UNK_TOKEN]
            token += 1
            
        return ids