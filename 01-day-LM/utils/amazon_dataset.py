import torch
from torchtext.data import TabularDataset 
from torch.utils.data import Dataset
from tqdm import tqdm_notebook, tqdm
_tqdm = tqdm_notebook
from operator import itemgetter 
import global_variables as gl

class AmazonDataset(Dataset):
    def __init__(self, data_list, max_inp_length=None, use_cuda=True):
        """
        data_list is a list of tuples: (x,y) where x is a list of ids and y is a label
        """
        self.data = data_list
        self.max_len = max_inp_length
        self.data_tensors = []
        device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        for d in tqdm(self.data):
            self.data_tensors.append((torch.LongTensor(d[:self.max_len]).to(device)))
                
    def __getitem__(self, key):
        inp = self.data_tensors[key]
        
        return inp#, len(inp)

    def __len__(self):
        return len(self.data)

def pad(tensor, length, dim=0, pad=0):
    """Pad tensor to a specific length.
    :param tensor: vector to pad
    :param length: new length
    :param dim: (default 0) dimension to pad
    :returns: padded tensor if the tensor is shorter than length
    """
    if tensor.size(dim) < length:
        return torch.cat(
            [tensor, tensor.new(*tensor.size()[:dim],
                                length - tensor.size(dim),
                                *tensor.size()[dim + 1:]).fill_(pad)],
            dim=dim)
    else:
        return tensor

def batchify(batch):
    data_list = []
    labels_list = []
    for b in batch:
        data_list.append(b[0])
        labels_list.append(b[1])
    data_batch = torch.stack(data_list, 0)
    labels_batch = torch.stack(labels_list, 0)

    return  data_batch, labels_batch


