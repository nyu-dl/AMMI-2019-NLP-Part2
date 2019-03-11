import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_agent import TorchAgent, Output
from torch import optim
import global_variables as gl

class BagOfNGrams(nn.Module):
    def init_layers(self):
        for l in self.layers:
            if getattr(l, 'weight', None) is not None:
                torch.nn.init.xavier_uniform_(l.weight)
    
    def __init__(self, vocab_size, emb_dim=300, hidden_size=256, out_size=128, reduce='sum', nlayers=2, activation='ReLU', dropout=0.1, batch_norm=False):
        super(BagOfNGrams, self).__init__()
       
        self.emb_dim = emb_dim
        self.reduce = reduce
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.activation = getattr(nn, activation)
        
        self.embedding = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=emb_dim, mode=reduce)
        if batch_norm is True:
            self.batch_norm = nn.BatchNorm1d(self.emb_dim)
        self.layers = nn.ModuleList([nn.Linear(self.emb_dim, self.hidden_size)])
        self.layers.append(self.activation())
        self.layers.append(nn.Dropout(p=dropout))
        
        for i in range(self.nlayers-2):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(self.activation())
            self.layers.append(nn.Dropout(p=dropout)) 
            
        self.layers.append(nn.Linear(self.hidden_size, self.out_size))
        self.init_layers()
    
    def forward(self, x):
        postemb = self.embedding(x)
        if hasattr(self, 'batch_norm'):
            x = self.batch_norm(postemb)
        else:
            x = postemb
        for l in self.layers:
            x = l(x)
        
        return x
    
class DecoderMLP(nn.Module):
    """Generates a token in response to context."""

    def __init__(self, input_size=128, output_size=1024, hidden_size=256):
        """Initialize decoder.
        :param input_size: size of embedding
        :param output_size: size of vocabulary
        :param hidden_size: size of the linear layers
        """
        super().__init__()
            
        self.linear = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        """Return encoded state.
        :param input: batch_size x 1 tensor of token indices.
        :param hidden: past (e.g. encoder) hidden state
        """
        output = F.relu(self.linear(input))
        scores = self.log_softmax(self.out(output))
        
        return scores 
    
class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, id2token, lr = 1e-3, use_cuda = True, 
                longest_label = 20, clip = 0.3, size_ngrams = 10):
        super(seq2seq, self).__init__()

        device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        self.device = device;
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        
        self.size_ngrams = size_ngrams
        self.id2token = id2token
        self.longest_label = longest_label

        # set up the criterion
        self.criterion = nn.NLLLoss()
        self.optims = {
             'nmt': optim.SGD(self.parameters(), lr=lr, nesterov=True, momentum = 0.99)
        }
        self.clip = clip

        self.START = torch.LongTensor([gl.SOS_IDX]).to(device)
        self.END_IDX = gl.EOS_IDX
        
    def save_model(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, filename)

    def load_model(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)        
        
    def zero_grad(self):
        """Zero out optimizer."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        for optimizer in self.optims.values():
            optimizer.step()
    
    def v2t(self, vector):
        return [self.id2token[i] for i in vector]
        
    def train_step(self, xs, ys):
        """Train model to produce ys given xs.
        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.
        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """
        if xs is None:
            return
        xs = xs.to(self.device)
        ys = ys.to(self.device)

        self.zero_grad()

        self.encoder.train()
        self.decoder.train()
            
        bow_output = self.encoder(xs)
        decoder_output = self.decoder(bow_output)
            
        loss = self.criterion(decoder_output, ys.view(-1))
        
        loss.backward()
        self.update_params()

        _max_score, predictions = decoder_output.max(1)
        
        return self.v2t(predictions), loss.item() 
    
    def eval_step(self, xs, ys):
        """Train model to produce ys given xs.
        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.
        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """
        if xs is None:
            return
        xs = xs.to(self.device)
        ys = ys.to(self.device)

        self.encoder.eval()
        self.decoder.eval()
            
        bow_output = self.encoder(xs)
        decoder_output = self.decoder(bow_output)
            
        loss = self.criterion(decoder_output, ys.view(-1))
        _max_score, predictions = decoder_output.max(1)
        
        return self.v2t(predictions), loss.item() 

    def evaluate(self, xs, use_context=False, score_only=False):
        """Generate a response to the input tokens.
        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.
        Return predicted responses (list of strings of length batchsize).
        """        
        if xs is None:
            return
        xs = xs.to(self.device)
        bsz = xs.size(0)
        ys = torch.cat((xs[0, 1:].unsqueeze(0), torch.LongTensor([[gl.EOS_IDX]])), dim=1)
    
        # just predict
        self.encoder.eval()
        self.decoder.eval()
        
        if score_only or not use_context:
            encoder_input = torch.LongTensor([gl.SOS_IDX] * self.size_ngrams)
            encoder_input = encoder_input.unsqueeze(0).repeat(bsz, 1)
        else:
            if xs.size(1) >= self.size_ngrams:
                encoder_input = xs[-self.size_ngrams:]
            else:
                encoder_input = torch.LongTensor([[gl.SOS_IDX] * (self.size_ngrams - xs.size(1))])
                encoder_input = torch.cat((encoder_input, xs), dim=1)   # this needs to be of shape bsz, self.size_ngrams
                
        predictions = []
        done = [False for _ in range(bsz)]
        total_done = 0
        scores = torch.zeros(bsz)
        score_counts = 0
        
        if score_only:
            num_predictions = xs.size(1)
        else:
            num_predictions = self.longest_label

        for i in range(num_predictions):
            decoder_input = self.encoder(encoder_input)
            decoder_output = self.decoder(decoder_input)
            
            loss = self.criterion(decoder_output, torch.LongTensor([ys[0][i]]))
            
            _max_score, next_token = decoder_output.max(1)
            
            scores = scores + loss.item()
            score_counts += 1
            
            if score_only:   # replace the next token with the one in the input data
                next_token = torch.index_select(xs, 1, torch.tensor([i])).squeeze(1)
                
            predictions.append(next_token)
            indices = torch.tensor([i for i in range(1, self.size_ngrams)])
            prev_tokens = torch.index_select(encoder_input, 1, indices)
            encoder_input = torch.cat((prev_tokens, next_token.unsqueeze(1)), 1)

            # stop if you've found the 
            for b in range(bsz):
                if not done[b]:
                    # only add more tokens for examples that aren't done
                    if next_token[b].item() == self.END_IDX:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
            if total_done == bsz:
                # no need to generate any more
                break
                
        predictions = [self.v2t(p) for p in predictions]
        scores = scores / score_counts
    
        return predictions, scores
    