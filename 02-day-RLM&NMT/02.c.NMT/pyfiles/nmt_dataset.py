import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset
import unicodedata
import re

from collections import namedtuple

import torch

from global_variables import SOS_IDX, SOS_TOKEN, EOS_IDX, EOS_TOKEN, UNK_IDX, UNK_TOKEN, PAD_IDX, PAD_TOKEN, device


class Lang:
	def __init__(self, name, minimum_count = 5):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = [None]*4
		self.index2word[SOS_IDX] = SOS_TOKEN
		self.index2word[EOS_IDX] = EOS_TOKEN
		self.index2word[UNK_IDX] = UNK_TOKEN
		self.index2word[PAD_IDX] = PAD_TOKEN

		self.word2count[SOS_TOKEN] = 100;
		self.word2count[EOS_TOKEN] = 100;
		self.word2count[UNK_TOKEN] = 100;
		self.word2count[PAD_TOKEN] = 100;

		self.word2index[SOS_TOKEN] = SOS_IDX;
		self.word2index[EOS_TOKEN] = EOS_IDX;
		self.word2index[UNK_TOKEN] = UNK_IDX;
		self.word2index[PAD_TOKEN] = PAD_IDX;
		self.n_words = 4  # Count SOS and EOS

		self.minimum_count = minimum_count;

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord( word.lower() )

	def addWord(self, word):
		if word not in self.word2count.keys():
			self.word2count[word] = 1
		else:
			self.word2count[word] += 1
			
		if self.word2count[word] >= self.minimum_count:
			if word not in self.index2word:
				word = str(word);
				self.word2index[word] = self.n_words
				self.index2word.append(word)
				self.n_words += 1

	def vec2txt(self, list_idx):
		word_list = []
		if type(list_idx) == list:
			for i in list_idx:
				if i not in [EOS_IDX, SOS_IDX, PAD_IDX]:
					word_list.append(self.index2word[i])
		else:
			for i in list_idx:
				if i.item() not in [EOS_IDX,SOS_IDX,PAD_IDX]:
					word_list.append(self.index2word[i.item()])
		return (' ').join(word_list)

	def txt2vec(self, sentence):
		token_list = sentence.lower().split();
		index_list = [self.word2index[token] if token in self.word2index else UNK_IDX for token in token_list]
		return torch.from_numpy(np.array(index_list)).to(device)


def read_dataset(file):
	f = open(file)
	list_l = []
	for line in f:
		list_l.append(line.strip())
	df = pd.DataFrame()
	df['data'] = list_l
	return df


def token2index_dataset(df, source_lang_obj, target_lang_obj):
	for lan in ['source','target']:
		indices_data = []
		if lan=='source':
			lang_obj = source_lang_obj
		else:
			lang_obj = target_lang_obj
			
		for tokens in df[lan+'_tokenized']:
			
			index_list = [lang_obj.word2index[token] if token in lang_obj.word2index else UNK_IDX for token in tokens]
			index_list.append(EOS_IDX)
			indices_data.append(index_list)
			
		df[lan+'_indized'] = indices_data
		
	return df

def load_or_create_language_obj(source_name, source_lang_obj_path, source_data, minimum_count):
	
	if not os.path.exists(source_lang_obj_path):
		os.makedirs(source_lang_obj_path)
	
	full_file_path = os.path.join(source_lang_obj_path, source_name+'_lang_obj_'+'min_count_'+str(minimum_count)+'.p')
	
	if os.path.isfile(full_file_path):
		source_lang_obj = pickle.load( open( full_file_path, "rb" ) );
	else:
		source_lang_obj = Lang(source_name, minimum_count);
		for i, line in enumerate(source_data):
#           if i%10000 == 0:
#               print(i, len(source_data))
#               print(str(float(i/len(source_data))*100)+' done');
			source_lang_obj.addSentence(line);
		pickle.dump( source_lang_obj, open(full_file_path , "wb" ) )
		
	return source_lang_obj


def load_language_pairs(filepath, source_name = 'en', target_name = 'vi',
						lang_obj_path = '.',  minimum_count = 5):

	source = read_dataset(filepath['source']);
	target = read_dataset(filepath['target']);
	
	main_df = pd.DataFrame();
	main_df['source_data'] = source['data'];
	main_df['target_data'] = target['data'];
	
	
	source_lang_obj = load_or_create_language_obj(source_name, lang_obj_path, main_df['source_data'], minimum_count);
	target_lang_obj = load_or_create_language_obj(target_name, lang_obj_path, main_df['target_data'], minimum_count);
	
	for x in ['source', 'target']:
		main_df[x+'_tokenized'] = main_df[x + "_data"].apply(lambda x:x.lower().split() );
		main_df[x+'_len'] = main_df[x+'_tokenized'].apply(lambda x: len(x)+1) #+1 for EOS
	
	main_df = token2index_dataset(main_df, source_lang_obj, target_lang_obj);
	
	# main_df = main_df[ np.logical_and( np.logical_and(main_df['source_len'] >=2, main_df['target_len'] >=2) , 
	#                               np.logical_and( main_df['source_len'] <= Max_Len, main_df['target_len'] <= Max_Len) ) ];

	main_df =  main_df[  np.logical_and(main_df['source_len'] >=2, main_df['target_len'] >=2 ) ]
	
	return main_df, source_lang_obj, target_lang_obj
	

class LanguagePair(Dataset):
	def __init__(self, source_name, target_name, filepath, 
					lang_obj_path, val = False, minimum_count = 5, max_num = None):

		##filepath is a dict with keys source and target
		
		self.source_name = source_name;
		self.target_name = target_name;
		self.val = val;
		self.minimum_count = minimum_count;

		self.main_df, self.source_lang_obj, self.target_lang_obj = load_language_pairs(filepath, 
																			  source_name, target_name, lang_obj_path, minimum_count);

		self.max_num = max_num;
		
	def __len__(self):
		return len( self.main_df ) if self.max_num is None else self.max_num
	
	def __getitem__(self, idx):
		
		return_list = [self.main_df.iloc[idx]['source_indized'], self.main_df.iloc[idx]['target_indized'], 
					self.main_df.iloc[idx]['source_len'], self.main_df.iloc[idx]['target_len'] ]

		if self.val:
			return_list.append(self.main_df.iloc[idx]['target_data'])
		
		return return_list 


def argsort(keys, *lists, descending=False):
    """Reorder each list in lists by the (descending) sorted order of keys.
    :param iter keys: Keys to order by.
    :param list[list] lists: Lists to reordered by keys's order.
                             Correctly handles lists and 1-D tensors.
    :param bool descending: Use descending order if true.
    :returns: The reordered items.
    """
    ind_sorted = sorted(range(len(keys)), key=lambda k: keys[k])
    if descending:
        ind_sorted = list(reversed(ind_sorted))
    output = []
    for lst in lists:
        if isinstance(lst, torch.Tensor):
            output.append(lst[ind_sorted])
        else:
            output.append([lst[i] for i in ind_sorted])
    return output


def vocab_collate_func(batch, MAX_LEN):
	source_data = []
	target_data = []
	source_len = []
	target_len = []

	for datum in batch:
		source_len.append(datum[2])
		target_len.append(datum[3])

	MAX_LEN_Source = np.min([ np.max(source_len), MAX_LEN ]);
	MAX_LEN_Target = np.min([np.max(target_len), MAX_LEN]);

	source_len = np.clip(source_len, a_min = None, a_max = MAX_LEN_Source )
	target_len = np.clip(target_len, a_min = None, a_max = MAX_LEN_Target )
	# padding
	for datum in batch:
		if datum[2]>MAX_LEN_Source:
			padded_vec_s1 = np.array(datum[0])[:MAX_LEN_Source]
		else:
			padded_vec_s1 = np.pad(np.array(datum[0]),
								pad_width=((0,MAX_LEN_Source - datum[2])),
								mode="constant", constant_values=PAD_IDX)
		if datum[3]>MAX_LEN_Target:
			padded_vec_s2 = np.array(datum[1])[:MAX_LEN_Target]
		else:
			padded_vec_s2 = np.pad(np.array(datum[1]),
								pad_width=((0,MAX_LEN_Target - datum[3])),
								mode="constant", constant_values=PAD_IDX)
		source_data.append(padded_vec_s1)
		target_data.append(padded_vec_s2)

	packed = True;
	if packed:
		source_data, source_len, target_data, target_len = argsort(source_len, source_data, source_len, target_data, target_len, descending=True)
	
	
	packed = False
	if packed:
		source_data, source_len, target_data, target_len = argsort(source_len, source_data, source_len, target_data, target_len, descending=True)

	named_returntuple = namedtuple('namedtuple', ['text_vecs', 'text_lens', 'label_vecs', 'label_lens', 'use_packed'])
	return_tuple =named_returntuple( torch.from_numpy(np.array(source_data)).to(device), 
									 torch.from_numpy(np.array(source_len)).to(device),
									 torch.from_numpy(np.array(target_data)).to(device),
									 torch.from_numpy(np.array(target_len)).to(device),
									 packed );

	return return_tuple
