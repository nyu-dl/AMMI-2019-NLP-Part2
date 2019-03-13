import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_IDX = 0
UNK_IDX = 1 
SOS_IDX = 2
EOS_IDX = 3
SEP_IDX = 4

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>' 
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
SEP_TOKEN = '<sep>'  # separates utterances in the dialogue history

NEAR_INF = 1e20  # approximates infinity for attention masking, scoring masking in beam
