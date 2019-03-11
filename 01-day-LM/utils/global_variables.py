import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNK_IDX = 0 
SOS_IDX = 1
EOS_IDX = 2

UNK_TOKEN = '<unk>' 
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
