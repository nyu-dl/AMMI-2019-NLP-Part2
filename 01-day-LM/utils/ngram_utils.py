import numpy as np
import spacy
import random
import numpy
import itertools
from operator import itemgetter 
from glob import glob
from tqdm import tqdm_notebook, tqdm
_tqdm = tqdm_notebook
from collections import Counter
import string
import re
import more_itertools as mit  # not built-in package
from collections import Counter
import re
import pandas
import altair
import pygtrie 
import global_variables as gl
import sys
import spacy
import math

# Load English tokenizer, tagger, parser, NER and word vectors
tokenizer = spacy.load('en_core_web_sm')               
punctuations = '"#$%&\()*+-/:;<=>@[\\]^_`{|}~'   # kept ' , . ? ! 

def lower_case_remove_punc(parsed):
    return [token.text.lower() for token in parsed if (token.text not in punctuations)]

def split_into_sentences(data):
    nlp = spacy.lang.en.English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    sentences = []
    for text in data:
        doc = nlp(text)
        for sent in doc.sents:
            sentences.append(sent.string)
    return sentences

def tokenize_dataset(dataset, batch_size=1024):
   # tokenize each sentence -- each tokenized sentence will be an element in token_dataset
    token_dataset = []
    # tokenize all words -- each token will be an item in all_tokens (in the order given by the list of sentences)
    all_tokens = []     # all the tokens -- 
    
    # split the reviews into sentences
    sentence_dataset = split_into_sentences(dataset) 
    # process data: make all the words lower case and remove some less relevant punctuation 
    for sample in _tqdm(tokenizer.pipe(sentence_dataset, disable=['parser', 'tagger', 'ner'], batch_size=batch_size, n_threads=1)):
        tokens = lower_case_remove_punc(sample) 
        token_dataset.append(tokens)    
        all_tokens += tokens

    return token_dataset, all_tokens

def pad_dataset(data, n=3):        
    result_list = []
    for l in data:
        padded = [gl.SOS_TOKEN for i in range(n - 1)] + l + [gl.EOS_TOKEN for i in range(n - 1)]
        result_list.append(padded)
    return result_list
    
def get_vocab(data, frac_vocab=0.9):
    all_train_tokens = list(mit.flatten(data))
    counted_tokens = Counter(all_train_tokens)
    max_vocab_size = int(frac_vocab * len(counted_tokens))
    vocab, _ = zip(*counted_tokens.most_common(max_vocab_size))
    
    return vocab

def get_dict(vocab):
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(3, 3+len(vocab)))) 
    id2token = [gl.UNK_TOKEN, gl.SOS_TOKEN, gl.EOS_TOKEN] + id2token

    token2id[gl.UNK_TOKEN] = gl.UNK_IDX
    token2id[gl.SOS_TOKEN] = gl.SOS_IDX 
    token2id[gl.EOS_TOKEN] = gl.EOS_IDX
    
    return id2token, token2id

def get_ids(data, token2id):
    data_ids = []
    for d in data:
        data_ids.append([token2id[t] if t in token2id else gl.UNK_IDX for t in d])
        
    return data_ids

    
class NgramLM:
    def __init__(self, tokenized_data, all_tokens, n=3, frac_vocab=0.9, \
                smoothing=None, delta=0.1, alpha=0.9):

        self.n = n
        self.frac_vocab = frac_vocab
        self.smoothing = smoothing
        self.alpha = alpha
        self.delta = delta
        self.raw_data = tokenized_data
        
        self.id2token, self.token2id = self.make_vocab_ids()
        self.padded_data = self.pad_sentences(self.n)
        self.ngram_data = self.find_ngrams(self.n)

        self.vocab_ngram, self.count_ngram = self.ngram_counts(self.n)
        self.vocab_unigram, self.count_unigram = self.ngram_counts(1)
        self.vocab_bigram, self.count_bigram = self.ngram_counts(2)
        self.vocab_trigram, self.count_trigram = self.ngram_counts(3)
        
        self.id2token_ngram, self.token2id_ngram = self.ngram_dict()
        
        self.trie_ngram = self.make_trie(self.n)
        self.trie_unigram = self.make_trie(1)
        self.trie_bigram = self.make_trie(2)
        self.trie_trigram = self.make_trie(3)
        
        if n > 1:
            self.vocab_prev_ngram, self.count_prev_ngram = self.ngram_counts(self.n - 1)
            self.trie_prev_ngram = self.make_trie(self.n - 1)
        else:
            self.vocab_prev_ngram, self.count_prev_ngram = None, None
            self.trie_prev_ngram = None
            
    def pad_sentences(self, n, sentence=None):
        if sentence:
            data = sentence
        else:
            data = self.raw_data
        result_list = []
        for l in data:
            padded = [gl.SOS_TOKEN for i in range(n - 1)] + l + [gl.EOS_TOKEN for i in range(n - 1)]
            result_list.append(padded)
        return result_list
    
    def make_vocab_ids(self):
        all_train_tokens = list(mit.flatten(self.raw_data))
        
        counted_tokens = Counter(all_train_tokens)
        max_vocab_size = int(self.frac_vocab * len(counted_tokens))

        vocab, _ = zip(*counted_tokens.most_common(max_vocab_size))
                        
        id2token = list(vocab)
        token2id = dict(zip(vocab, range(3, 3+len(vocab)))) 
        id2token = [gl.UNK_TOKEN, gl.SOS_TOKEN, gl.EOS_TOKEN] + id2token

        token2id[gl.UNK_TOKEN] = gl.UNK_IDX
        token2id[gl.SOS_TOKEN] = gl.SOS_IDX 
        token2id[gl.EOS_TOKEN] = gl.EOS_IDX

        return id2token, token2id
    
    def find_ngrams(self, n, sentence=None):
        if sentence:
            data = sentence
        else:
            data = self.pad_sentences(n)
        result_list = []
        for l in data:
            result_list.append(list(zip(*[l[i:] for i in range(n)])))
        return result_list

    def ngram_counts(self, n=None): 
        if n == None:
            n = self.n
        ngram_data = self.find_ngrams(n)
        all_train_tokens = list(mit.flatten(ngram_data))
        
        counted_tokens = Counter(all_train_tokens)
        max_vocab_size = int(self.frac_vocab * len(counted_tokens))

        vocab, count = zip(*counted_tokens.most_common(max_vocab_size))
        
        return vocab, count
    
    def ngram_dict(self, vocab=None):
        if vocab == None:
            vocab = self.vocab_ngram
            
        id2token_ngram = list(vocab)
        token2id_ngram = dict(zip(vocab, range(len(vocab)))) 

        return id2token_ngram, token2id_ngram
    
    def convert_to_trie(self, x):
        return '/'.join(x) 
        
    def convert_to_ngram(self, x):
        return tuple(x.split('/'))
        
    # TODO: the trie needs to be made out of the entire dict and using ids not tokens
    def make_trie(self, n=None):
        if n == None:
            n = self.n
            
        trie = pygtrie.StringTrie()
        for vn, cn in zip(self.vocab_ngram, self.count_ngram):
            tn = self.convert_to_trie(vn)
            if tn not in trie:
                trie[tn] = cn
        return trie 
    
    def get_ngram_count(self, ngram, trie=None):
        if trie is None:
            trie = self.trie_ngram
        nt = self.convert_to_trie(ngram)
        if nt in trie:
            return trie[nt]
        else:
            return 0
        
    def get_ngram_prob(self, ngram):
        if self.smoothing == None:
            c = self.get_ngram_count(ngram)
            nt_prefix = self.convert_to_trie(ngram[:-1])
            all_counts = 0
            if self.trie_ngram.has_subtrie(nt_prefix):
                prefixes = self.trie_ngram.items(prefix=nt_prefix)
    #             print(nt_prefix, prefixes, "\n")
                all_counts = sum([prefixes[i][1] for i in range(len(prefixes))])
            if all_counts > 0:
                return c / all_counts
            else:
                return 0
            
        elif self.smoothing == 'additive':
            return self.get_ngram_prob_additive_smoothing(ngram, delta=self.delta)
        
        elif self.smoothing == 'add-one':
            return self.get_ngram_prob_add_one_smoothing(ngram)
        
        elif self.smoothing == 'interpolation':
            return self.get_ngram_prob_interpolation_smoothing(ngram, alpha=self.alpha)
            
        elif self.smoothing == 'discounting':
            return self.get_p_bi(ngram[-1], ngram[:-1])

    def get_ngram_prob_additive_smoothing(self, ngram, delta=0.5):
        c = self.get_ngram_count(ngram) + delta*1
        nt_prefix=  self.convert_to_trie(ngram[:-1])
        all_counts = 0
        if self.trie_ngram.has_subtrie(nt_prefix):  # check if prefix exists in trie
            prefixes = self.trie_ngram.items(prefix=nt_prefix)
            all_counts = sum([prefixes[i][1] for i in range(len(prefixes))])
        all_counts += delta*len(self.id2token)
        if all_counts > 0:
            return c / all_counts
        else:
            return 0 
        
    def get_ngram_prob_add_one_smoothing(self, ngram):
        return self.get_ngram_prob_additive_smoothing(ngram, delta=1)

    def get_ngram_prob_interpolation_smoothing(self, ngram, alpha=0.8):
        c = self.get_ngram_count(ngram, trie=self.trie_ngram)
        prefix=  self.convert_to_trie(ngram[:-1])
        all_counts = 0
        if self.trie_ngram.has_subtrie(prefix):
            prefixes = self.trie_ngram.items(prefix=prefix)
            all_counts = sum([prefixes[i][1] for i in range(len(prefixes))])
        if all_counts > 0:
            prob_ngram = c / all_counts
        else:
            prob_ngram = 0
            
        prev_ngram = tuple(list(ngram[1:]))
        prev_c = self.get_ngram_count(prev_ngram, trie=self.trie_prev_ngram)
        prev_prefix=  self.convert_to_trie(prev_ngram[:-1])
        prev_all_counts = 0
        if prev_prefix in self.trie_prev_ngram:
            prev_prefixes = self.trie_prev_ngram.items(prefix=prev_prefix)
            prev_all_counts = sum([prev_prefixes[i][1] for i in range(len(prev_prefixes))])
        if prev_all_counts > 0:
            prob_prev_ngram = prev_c  / prev_all_counts
        else: 
            prob_prev_ngram = 0
            
        return alpha*(prob_ngram) + (1-alpha)*prob_prev_ngram

    def get_unigram_count(self, r):
        return np.sum([1 for t in self.trie_unigram if self.trie_unigram[t] == r])                

    def get_bigram_count(self, r):
        return np.sum([1 for t in self.trie_bigram if self.trie_bigram[t] == r])

    def get_biunigram_count(self, r, token):
        counts = 0
        prefix = self.convert_to_trie(token)   # token needs to be a single token 
        if self.trie_bigram.has_subtrie(prefix):
            prefixes = self.trie_bigram.items(prefix=prefix)
            counts = sum([1 for i in range(len(prefixes)) if prefixes[i][1] == r ]) 
        return counts

    def get_b_bi(self):
        bbi = self.get_bigram_count(1) / (self.get_bigram_count(1) + 2 * self.get_bigram_count(2))
        return bbi

    def get_b_uni(self):
        buni = self.get_unigram_count(1) / (self.get_unigram_count(1) + 2 * self.get_unigram_count(2))
        return buni

    # TODO: normalize this pd
    def get_p_uni(self, w):
        N = len(self.token2id)
        
        uni_w = self.convert_to_trie(w)
        if uni_w in self.trie_unigram:
            N_w = self.trie_unigram[uni_w]
        else:
            N_w = sys.float_info.min

        b_uni = self.get_b_uni()

        W = len(self.id2token)
        N_0 = self.get_unigram_count(0)
        
        p_uni = max((N_w - b_uni / N), 0) + b_uni * (W - N_0) / N * 1 / W

        return p_uni

    def get_p_bi(self, w, v):   # w given v
        bigram = self.convert_to_trie(tuple([v] + [w]))
        if bigram in self.trie_bigram:
            N_vw = self.trie_bigram[bigram]
        else:
            N_vw = sys.float_info.min

        uni_v = self.convert_to_trie(tuple([v]))
        if uni_v in self.trie_unigram:
            N_v = self.trie_unigram[uni_v]
        else:
            N_v = sys.float_info.min
            
        b_bi = self.get_b_bi()
        b_uni = self.get_b_uni()

        p_uni = self.get_p_uni(tuple([w]))

        W = len(self.id2token)
        N_0 = self.get_biunigram_count(0, v)


        p_bi =  max((N_vw - b_bi) / N_v,  0) + \
             b_bi * (W - N_0) / N_v * p_uni

        return p_bi

    def get_prob_sentence(self, sentence):
        padded_sentence = self.pad_sentences(self.n, sentence=sentence)  # needs a list
        ngram_sentence = self.find_ngrams(self.n, sentence=padded_sentence)[0] # only one element in list
        prob = 1
        for ngram in ngram_sentence:
            prob_ngram = self.get_ngram_prob(ngram)
            prob *= prob_ngram
        return prob
    
    def get_score_sentence(self, sentence):
        padded_sentence = self.pad_sentences(self.n, sentence=sentence)  # needs a list
        ngram_sentence = self.find_ngrams(self.n, sentence=padded_sentence)[0] # only one element in list
        score = 0
        count = 0
        for ngram in ngram_sentence:
            prob_ngram = self.get_ngram_prob(ngram)
            if prob_ngram > 0:
                score += np.log(prob_ngram)
            else:
                score += np.log(sys.float_info.min)    
            count += 1
        ppl = math.exp(- score / (count + 1))
        return ppl
    
    def get_prob_distr_ngram(self, prev_tokens, smoothing=None):
        pd = [0 for v in self.id2token]
        nt_prefix = self.convert_to_trie(prev_tokens)
        if self.trie_ngram.has_subtrie(nt_prefix):
            prefixed_ngrams = self.trie_ngram.items(prefix=nt_prefix)
            for ngram in prefixed_ngrams:
                suffix = ngram[0].split('/')[-1]      # get the suffix of this ngram
                if suffix in self.token2id:
                    idx_suffix = self.token2id[suffix]     # get the idx in the vocabulary of this suffix token
                else:
                    idx_suffix = self.token2id[gl.UNK_TOKEN]
                pd[idx_suffix] = self.get_ngram_prob(self.convert_to_ngram(ngram[0]))
        if sum(pd) > 0:
            return [p/sum(pd) for p in pd]
        else:
            return [1/len(pd) for p in pd]
        
    def sample_from_pd(self, prev_tokens):
        pd = self.get_prob_distr_ngram(prev_tokens)
        idx_next_token = np.random.choice(len(self.id2token), 1, p=pd)[0]
        return self.id2token[idx_next_token]

    def generate_sentence(self, num_tokens, context=None):
        sentence = []
        if context is None:
            prev_tokens = tuple([gl.SOS_TOKEN] * (self.n - 1))
        else:
            if len(context) >= self.n - 1:
                prev_tokens = context[-(self.n - 1):]
            else:
                prev_tokens = tuple([gl.SOS_TOKEN] * (self.n - 1 - len(context)) + [c for c in context])
        
        for i in range(num_tokens):
            next_token = self.sample_from_pd(prev_tokens)
            prev_tokens = tuple(list(prev_tokens[1:]) + [next_token])
            sentence.append(next_token)
            print(' '.join(sentence))
            if next_token == '<eos>':
                break
                
        return ' '.join(sentence)
    
    def get_perplexity(self, test_sentences, subsample=None):
        ll = 0
        num_tokens = 0
        for i, s in enumerate(test_sentences):
            if subsample and i % subsample == 0:
                prob = self.get_prob_sentence([s])
                ll += np.log(prob + sys.float_info.min)
                num_tokens += len(s) + 1
        ppl = np.exp(-ll/num_tokens)
        return ppl

    def _text2id(self, doc):
        return [self.token2id[t] if t in self.token2id else gl.UNK_IDX for t in doc]

    def _id2text(self, vec):
        return [self.id2token[i] for i in vec]
