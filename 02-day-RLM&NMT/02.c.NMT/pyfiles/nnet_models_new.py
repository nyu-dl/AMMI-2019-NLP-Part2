import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import global_variables
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

import bleu_score

PAD_IDX = global_variables.PAD_IDX;
SOS_IDX = global_variables.SOS_IDX;
EOS_IDX = global_variables.EOS_IDX;

class BagOfWords(nn.Module):
    def init_layers(self):
        for l in self.layers:
            if getattr(l, "weight", None) is not None:
                torch.nn.init.xavier_uniform_(l.weight)

    def __init__(
        self,
        input_size,
        hidden_size=512,
        reduce="sum",
        nlayers=2,
        activation="ReLU",
        dropout=0.1,
        batch_norm=False,
    ):
        super(BagOfWords, self).__init__()

        self.emb_dim = hidden_size

        self.reduce = reduce
        assert(self.reduce in ["sum", "mean", "max"]);

        self.nlayers = nlayers
        self.hidden_size = hidden_size

        self.activation = getattr(nn, activation)

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx = PAD_IDX)

        if batch_norm is True:
            self.batch_norm = nn.BatchNorm1d(self.emb_dim)
        self.layers = nn.ModuleList([nn.Linear(self.emb_dim, self.hidden_size)])

        self.layers.append(self.activation())
        self.layers.append(nn.Dropout(p=dropout))
        for i in range(self.nlayers - 2):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(self.activation())
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.init_layers()

    def forward(self, x):
        postemb = self.embedding(x)

        if self.reduce == "sum":
            postemb = postemb.sum(dim=1);
        elif self.reduce == "mean":
            postemb = postemb.mean(dim=1);
        elif self.reduce == "max":
            postemb = postemb.max(dim=1)[0];

        if hasattr(self, "batch_norm"):
            x = self.batch_norm(postemb)
        else:
            x = postemb

        for l in self.layers:
            x = l(x)

        return None, x.unsqueeze(0)


class EncoderRNN(nn.Module):
    """Encodes the input context."""

    def __init__(self, input_size, hidden_size, numlayers):
        """Initialize encoder.
        :param input_size: size of embedding
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx = PAD_IDX)
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=numlayers, batch_first=True
        )

    def forward(self, input, hidden=None):
        """Return encoded state.
        :param input: (batchsize x seqlen) tensor of token indices.
        :param hidden: optional past hidden state
        """
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    """Generates a sequence of tokens in response to context."""

    def __init__(self, output_size, hidden_size, numlayers):
        """Initialize decoder.
        :param input_size: size of embedding
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=numlayers, batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, encoder_output=None, xs_len = None, context_vec = None):
        """Return encoded state.
        :param input: batch_size x 1 tensor of token indices.
        :param hidden: past (e.g. encoder) hidden state
        """
        emb = self.embedding(input)
        rel = F.relu(emb)
        output, hidden = self.gru(rel, hidden)
        scores = self.softmax(self.out(output))
        return scores, hidden, None, None


class Attention_Module(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Attention_Module, self).__init__()
        self.l1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim + output_dim, output_dim, bias=False)

    def forward(self, hidden, encoder_outs, src_lens):
        ''' hiddden: bsz x hidden_dim
        encoder_outs: bsz x sq_len x encoder dim (output_dim)
        src_lens: bsz

        x: bsz x output_dim
        attn_score: bsz x sq_len'''

        x = self.l1(hidden)
#         att_score = (encoder_outs.transpose(0, 1) * x.unsqueeze(0)).sum(dim=2)
        att_score = torch.bmm(encoder_outs, x.unsqueeze(-1)); #this is bsz x seq x 1
        att_score = att_score.squeeze(-1); #this is bsz x seq
        att_score = att_score.transpose(0, 1);
        
        seq_mask = self.sequence_mask(src_lens, 
                                    max_len=max(src_lens).item(), 
                                    device = hidden.device).transpose(0, 1)


        masked_att = seq_mask * att_score
        masked_att[masked_att == 0] = -1e10
        attn_scores = F.softmax(masked_att, dim=0)
        x = (attn_scores.unsqueeze(2) * encoder_outs.transpose(0, 1)).sum(dim=0)
        x = torch.tanh(self.l2(torch.cat((x, hidden), dim=1)))
        return x, attn_scores

    def sequence_mask(self, sequence_length, max_len=None, device = torch.device('cuda')):
        if max_len is None:
            max_len = sequence_length.max().item()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).repeat([batch_size, 1])
        seq_range_expand = seq_range_expand.to(device)
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        return (seq_range_expand < seq_length_expand).float()


class Decoder_SelfAttn(nn.Module):
    """Generates a sequence of tokens in response to context with self attention.
       Note that this is the same as previous decoder if self_attention=False"""


    def __init__(self, output_size, hidden_size, idropout=0.5, self_attention = False, encoder_attention = False):
        super(Decoder_SelfAttn, self).__init__()

        self.output_size = output_size;

        self.self_attention = self_attention;
        self.encoder_attention = encoder_attention;

        self.hidden_size = hidden_size;
        self.embedding = nn.Embedding(output_size, hidden_size);

        self.memory_rnn = nn.GRUCell(hidden_size + int(self.encoder_attention==True)*self.hidden_size, 
                                    hidden_size, bias=True);

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        if self.self_attention:
            self.projector_summ = nn.Sequential(nn.Dropout(idropout),
                                                nn.Linear(hidden_size*2, hidden_size),
                                                nn.Dropout(idropout))

        if self.encoder_attention:
            self.encoder_attention_module = Attention_Module(self.hidden_size, self.hidden_size);
        
    def forward(self, input, memory, encoder_output = None, xs_len = None, context_vec = None):
        memory = memory.transpose(0, 1);
        emb = self.embedding(input)
        emb = F.relu(emb)
        
        
        emb = emb.transpose(0, 1);
        return_scores = torch.empty(emb.size(0), emb.size(1), self.output_size).to(input.device)
        
        if context_vec is None and self.encoder_attention:
            context_vec = torch.zeros([emb.size(1), self.hidden_size]).to(emb.device);

        if self.encoder_attention:
            attn_wts_list = [];
        else:
            attn_wts_list = None;

        for t in range(emb.size(0)):
            current_vec = emb[t];
            
            if self.self_attention:
                selected_memory, attention0 = self.calculate_self_attention(current_vec, memory)

            elif self.encoder_attention:
                current_vec = torch.cat([current_vec, context_vec], dim = 1);
                selected_memory = memory[:, 0, :];

                
            if ( not (self.self_attention or self.encoder_attention)):    
                selected_memory, attention0 = memory[:, 0, :], None;

            # recurrent
            mem_out = self.memory_rnn(current_vec, selected_memory);
    
            if self.encoder_attention:
                context_vec, attention0 = self.encoder_attention_module(mem_out, encoder_output, xs_len);
                scores = self.out(context_vec);
                attn_wts_list.append(attention0)
            else:
                scores = self.out(mem_out)

            scores = self.softmax(scores);
            return_scores[t] = scores

            if self.self_attention:
                 # update memory
                memory = torch.cat([mem_out[:, None, :], memory[:, :-1, :]], dim=1);
            else:
                memory = mem_out[:, None, :];
            
        return return_scores.transpose(0, 1).contiguous(), memory.transpose(0,1), attn_wts_list, context_vec

    def calculate_self_attention(self, input, memory):
        # select memory to use
        concat_vec = torch.cat([input,  memory[:, 0, :]], dim=1);
        projected_vec = self.projector_summ(concat_vec);
    
        dot_product_values = torch.bmm(memory, projected_vec.unsqueeze(-1)).squeeze(-1)/ math.sqrt(self.hidden_size);
        
        weights =  F.softmax(dot_product_values, dim = 1).unsqueeze(-1);
        
        selected_memory = torch.sum( memory * weights, dim=1)
        return selected_memory, weights



class seq2seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        lr=1e-3,
        use_cuda=True,
        hiddensize=128,
        numlayers=2,
        target_lang=None,
        longest_label=20,
        clip=0.3,
    ):
        super(seq2seq, self).__init__()

        device = torch.device(
            "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"
        )
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        self.target_lang = target_lang

        # set up the criterion
        self.criterion = nn.NLLLoss()

        # set up optims for each module
        # self.optims = {
        #     'encoder': optim.SGD(encoder.parameters(), lr=lr, nesterov=True, momentum = 0.99),
        #     'decoder': optim.SGD(decoder.parameters(), lr=lr, nesterov=True, momentum = 0.99)
        # }

        self.optims = {
            "nmt": optim.SGD(self.parameters(), lr=lr, nesterov=True, momentum=0.99)
        }

        self.scheduler = {}
        for x in self.optims.keys():
            self.scheduler[x] = ReduceLROnPlateau(
                self.optims[x], mode="max", min_lr=1e-4, patience=0, verbose=True
            )

        self.longest_label = longest_label
        self.hiddensize = hiddensize
        self.numlayers = numlayers
        self.clip = clip
        self.START = torch.LongTensor([SOS_IDX]).to(device)
        self.END_IDX = EOS_IDX

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

    def scheduler_step(self, val_bleu):
        for scheduler in self.scheduler.values():
            scheduler.step(val_bleu)

    def v2t(self, vector):
        """Convert vector to text.
        :param vector: tensor of token indices.
            1-d tensors will return a string, 2-d will return a list of strings
        """
        if vector.dim() == 1:
            output_tokens = []
            # Remove the final END_TOKEN that is appended to predictions
            for token in vector:
                if token == self.END_IDX:
                    break
                else:
                    output_tokens.append(token)
            return self.target_lang.vec2txt(output_tokens)

        elif vector.dim() == 2:
            return [self.v2t(vector[i]) for i in range(vector.size(0))]
        raise RuntimeError(
            "Improper input to v2t with dimensions {}".format(vector.size())
        )

    def get_bleu_score(self, val_loader):
        
        bl = bleu_score.BLEU_SCORE();
        predicted_list = []
        real_list = []

        for data in val_loader:
            predicted_list += self.eval_step(data)
            real_list += self.v2t(data.label_vecs)

        return bl.corpus_bleu(predicted_list, [real_list])[0]

    def train_step(self, batch):
        """Train model to produce ys given xs.
        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.
        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """
        xs, xs_len, ys = batch.text_vecs, batch.text_lens, batch.label_vecs

        if xs is None:
            return
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        xs_len = xs_len.to(self.device)

        bsz = xs.size(0)
        starts = self.START.expand(bsz, 1)  # expand to batch size
        loss = 0
        self.zero_grad()
        self.encoder.train()
        self.decoder.train()
        target_length = ys.size(1)
        # save largest seen label for later
        self.longest_label = max(target_length, self.longest_label)

        encoder_output, encoder_hidden = self.encoder(xs)

        # Teacher forcing: Feed the target as the next input
        y_in = ys.narrow(1, 0, ys.size(1) - 1)
        decoder_input = torch.cat([starts, y_in], 1)

        decoder_output, decoder_hidden, _, _ = self.decoder(decoder_input,
                                                      encoder_hidden, 
                                                      encoder_output,
                                                      xs_len)


        scores = decoder_output.view(-1, decoder_output.size(-1))
        loss = self.criterion(scores, ys.view(-1))
        loss.backward()
        self.update_params()

        _max_score, predictions = decoder_output.max(2)
        return self.v2t(predictions), loss.item()

    def eval_step(self, batch, return_attn = False):
        """Generate a response to the input tokens.
        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.
        Return predicted responses (list of strings of length batchsize).
        """
        xs, xs_len = batch.text_vecs, batch.text_lens

        if xs is None:
            return

        xs = xs.to(self.device)
        xs_len = xs_len.to(self.device)

        bsz = xs.size(0)
        starts = self.START.expand(bsz, 1)  # expand to batch size
        # just predict
        self.encoder.eval()
        self.decoder.eval()
        encoder_output, encoder_hidden = self.encoder(xs)

        predictions = []
        done = [False for _ in range(bsz)]
        total_done = 0
        decoder_input = starts
        decoder_hidden = encoder_hidden
        
        attn_wts_list = []
        context_vec = None;

        for i in range(self.longest_label):
            # generate at most longest_label tokens

            decoder_output, decoder_hidden, attn_wts, context_vec = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_output, 
                                                          xs_len, 
                                                          context_vec)

            _max_score, preds = decoder_output.max(2)
            predictions.append(preds)
            decoder_input = preds  # set input to next step
            
            attn_wts_list.append(attn_wts)

            # check if we've produced the end token
            for b in range(bsz):
                if not done[b]:
                    # only add more tokens for examples that aren't done
                    if preds[b].item() == self.END_IDX:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
            if total_done == bsz:
                # no need to generate any more
                break
        predictions = torch.cat(predictions, 1)
        
        if return_attn:
            return self.v2t(predictions), attn_wts_list
        return self.v2t(predictions)
