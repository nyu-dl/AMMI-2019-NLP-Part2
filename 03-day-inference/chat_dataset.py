import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter, namedtuple
from tqdm import tqdm
from global_variables import SOS_IDX, SOS_TOKEN, EOS_IDX, EOS_TOKEN, UNK_IDX, UNK_TOKEN, PAD_IDX, PAD_TOKEN, SEP_IDX, SEP_TOKEN, device

RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)

Batch = namedtuple('Batch', ['text_vecs', 'text_lens', 'label_vecs', 'label_lens', 'use_packed'])

def stop_comprehension():
    raise StopIteration

class TextDataset(Dataset):
    """
    Simple text dataset. Loads everything in RAM. Preprocess all data in tensors
    in advance.

    """
    def __init__(self, text_file_path, max_voc_size=None, device='cpu', dictionary=None):
        """
        :param text_file_path: filename of the dataset file
        :param max_voc_size: max number of words in the dictionary
        """

        self.device = device
        text_data = open(text_file_path, 'r').readlines()
        
        
        if dictionary is None:
            self.history_starts = []
            self.input_text = []
            self.target_text = []
            self.word2ind = {}  # word:index
            self.ind2word = {}  # index:word
            self.counts = {}  # word:count
            
            self.word2ind[SOS_TOKEN] = SOS_IDX
            self.word2ind[EOS_TOKEN] = EOS_IDX
            self.word2ind[UNK_TOKEN] = UNK_IDX
            self.word2ind[PAD_TOKEN] = PAD_IDX
            self.word2ind[SEP_TOKEN] = SEP_IDX
            # high counts to avoid truncation due to max_voc_size
            self.counts['<sos>'] = 1e10
            self.counts['<eos>'] = 1e10
            self.counts['<unk>'] = 1e10
            self.counts['<pad>'] = 1e10
            self.counts['<sep>'] = 1e10
            
            for i, line in tqdm(enumerate(text_data), total=len(text_data)):
                _line = line.split('\t')
                _history_starts = int(_line[0])
                self.history_starts.append(_history_starts)
                _input_text = RETOK.findall(_line[1])
                _target_text = RETOK.findall(_line[2].rstrip())
               

                for word in _input_text+_target_text:
                    if word in self.word2ind.keys():
                        self.counts[word] += 1
                    else:
                        self.word2ind[word] = len(self.word2ind)
                        self.counts[word] = 1

                #import ipdb; ipdb.set_trace()
                self.input_text.append(_input_text)
                self.target_text.append(_target_text)
                
            self.ind2word = {v:k for k,v in self.word2ind.items()}
            self.counts = {k:v for (k,v) in Counter(self.counts).most_common(max_voc_size)}
            
        else:
            self.input_text = []
            self.target_text = []
            self.history_starts = []
            self.word2ind = dictionary['word2ind']
            self.ind2word = dictionary['ind2word']
            self.counts = dictionary['counts']
            text_data = open(text_file_path, 'r').readlines()
            for i, line in tqdm(enumerate(text_data), total=len(text_data)):
                _line = line.split('\t')
                _history_starts = int(_line[0])
                self.history_starts.append(_history_starts)
                _input_text = RETOK.findall(_line[1])
                _target_text = RETOK.findall(_line[2].rstrip())
                
                self.input_text.append(_input_text)
                self.target_text.append(_target_text)
                
        self.shared_dict = {'word2ind': self.word2ind, 'ind2word': self.ind2word, 'counts': self.counts}

    def t2v(self, tokenized_text):
        return [self.word2ind[w] if w in self.counts else self.word2ind[UNK_TOKEN] for w in tokenized_text]

    def v2t(self, list_ids):
        return ' '.join([self.ind2word[i] for i in list_ids])
    
    def pred2text(self, tensor):
        result = []
        for i in range(tensor.size(0)):
            if tensor[i].item() == EOS_IDX  or tensor[i].item() == PAD_IDX:
                break
            else:
                result.append(self.ind2word[tensor[i].item()])
        return ' '.join(result)
                
    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        _history_starts = self.history_starts[idx]
        _history = []
        for i in range(_history_starts, idx):
            _history.extend(self.input_text[i])
            _history.append(SEP_TOKEN)
            _history.extend(self.target_text[i])
            _history.append(SEP_TOKEN)
        return torch.Tensor(self.t2v(_history + self.input_text[idx])).long().to(self.device), torch.Tensor(self.t2v(self.target_text[idx])+[EOS_IDX]).long().to(self.device)

    def get_vocab_size(self):
        return len(self.counts)


def pad_tensor(tensors, sort=True):
    rows = len(tensors)
    lengths = [len(i) for i in tensors]
    max_t = max(lengths)
        
    output = tensors[0].new(rows, max_t)
    output.fill_(PAD_IDX)
    for i, (tensor, length) in enumerate(zip(tensors, lengths)):
        output[i,:length] = tensor

    return output, lengths

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

def batchify(batch):
    inputs = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    
    input_vecs, input_lens = pad_tensor(inputs)
    label_vecs, label_lens = pad_tensor(labels)
    # sort only wrt inputs here for encoder packinng
    input_vecs, input_lens, label_vecs, label_lens = argsort(input_lens, input_vecs, input_lens, label_vecs, label_lens, descending=True)

    return Batch(text_vecs=input_vecs, text_lens=input_lens, label_vecs=label_vecs, label_lens=label_lens, use_packed=True)
