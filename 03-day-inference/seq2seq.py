from beam import Beam
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from global_variables import SOS_IDX, SOS_TOKEN, EOS_IDX, EOS_TOKEN, UNK_IDX, UNK_TOKEN, PAD_IDX, PAD_TOKEN, SEP_IDX, SEP_TOKEN, device, NEAR_INF
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import pdb



class EncoderRNN(nn.Module):
    """Encodes the input context."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, pad_idx, dropout=0, shared_lt=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.pad_idx = pad_idx
        
        if shared_lt is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size, pad_idx)
        else:
            self.embedding = shared_lt
            
        self.gru = nn.GRU(
            self.embed_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        
        
    def forward(self, text_vec, text_lens, hidden=None, use_packed=True):
        embedded = self.embedding(text_vec)
        attention_mask = text_vec.ne(self.pad_idx)

        embedded = self.dropout(embedded)
        if use_packed is True:
            embedded = pack_padded_sequence(embedded, text_lens, batch_first=True)
        output, hidden = self.gru(embedded, hidden)
        if use_packed is True:
            output, output_lens = pad_packed_sequence(output, batch_first=True)
        
        return output, hidden, attention_mask


class AttentionLayer(nn.Module):

    def __init__(self, hidden_size, embedding_size, attention_type='general', attention_time='post'):
        super().__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_time = attention_time
        self.attention_type = attention_type
        if self.attention_time == 'pre':
            input_dim = embedding_size
        else:
            input_dim = hidden_size

        if self.attention_type == 'general':
            self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)

        self.linear_out = nn.Linear(hidden_size+input_dim, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, decoder_output, decoder_hidden, encoder_output, attention_mask):


        batch_size, seq_length, hidden_size = encoder_output.size()
        last_hidden_layer = decoder_hidden[-1].unsqueeze(1)

        encoder_output_t = encoder_output.transpose(1,2)


        if self.attention_type == 'general':
            hid = self.linear_in(last_hidden_layer)
        else:
            hid = last_hidden_layer

        attention_scores = torch.bmm(hid, encoder_output_t).squeeze(1)

        attention_scores.masked_fill_((1 - attention_mask), -NEAR_INF)
        attention_weights = self.softmax(attention_scores)

        mix = torch.bmm(attention_weights.unsqueeze(1), encoder_output)

        combined = torch.cat((decoder_output.squeeze(1), mix.squeeze(1)), dim=1)

        output = self.linear_out(combined).unsqueeze(1)
        output = self.tanh(output)

        return output, attention_weights


class DecoderRNN(nn.Module):
    """Generates a sequence of tokens in response to context."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, pad_idx, dropout=0, attention_type='general', attention_time='post'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, pad_idx)
        self.gru = nn.GRU(
            self.embed_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = AttentionLayer(self.hidden_size, embed_size,  attention_type, attention_time) if attention_type is not False else None;
        self.attention_time = attention_time

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, text_vec, decoder_hidden, encoder_states):
        emb = self.embedding(text_vec)
        emb = self.dropout(emb)
        seqlen = text_vec.size(1)
        encoder_output, encoder_hidden, attention_mask = encoder_states
        
        decoder_hidden = decoder_hidden
        output = []
        attn_w_log = []

        if self.attention_time == 'pre':
            attended_inputs = []
            for i in range(seqlen):

                attended_input, attention_weights = self.attention(emb[:,i:i+1], decoder_hidden, encoder_output, attention_mask)
                attended_inputs.append(attended_input)
                attn_w_log.append(attention_weights)
            emb = torch.cat(attended_inputs, 1).to(emb.device)



        for i in range(seqlen):
            decoder_output, decoder_hidden = self.gru(emb[:,i,:].unsqueeze(1), decoder_hidden)
            if self.attention_time == 'post':
                decoder_output_attended, attn_weights = self.attention(decoder_output, decoder_hidden, encoder_output, attention_mask)
                output.append(decoder_output_attended)
                attn_w_log.append(attn_weights)
            else:
                output.append(decoder_output);
            
        output = torch.cat(output, dim=1).to(text_vec.device)
        scores = self.out(output)       
        return scores, decoder_hidden, attn_w_log


class seq2seq(nn.Module):
    """
    Generic seq2seq model with attention mechanism.
    """
    def __init__(self, vocab_size_encoder, vocab_size_decoder, embedding_size, encoder_type='rnn', hidden_size=64, num_layers=2, lr=0.01, 
                       pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX, encoder_shared_lt=False, dropout=0.0, use_cuda=True, optimizer='Adam', 
                       grad_clip=None, encoder_attention = 'general', attention_time='post'):

        super().__init__()
        self.opts = {}
        self.opts['vocab_size_encoder'] = vocab_size_encoder
        self.opts['vocab_size_decoder'] = vocab_size_decoder
        self.opts['hidden_size'] = hidden_size
        self.opts['device'] = 'cuda' if use_cuda is True else 'cpu'
        self.opts['embedding_size'] = embedding_size
        self.opts['encoder_type'] = encoder_type
        self.opts['num_layers'] = num_layers
        self.opts['lr'] = lr
        self.opts['pad_idx'] = pad_idx
        self.opts['sos_idx'] = sos_idx
        self.opts['eos_idx'] = eos_idx
        self.opts['dropout'] = dropout
        self.opts['encoder_shared_lt'] = encoder_shared_lt
        self.opts['grad_clip'] = grad_clip
        self.opts['encoder_attention'] = encoder_attention;
        self.opts['attention_time'] = attention_time

        
        self.decoder = DecoderRNN(self.opts['vocab_size_decoder'], self.opts['embedding_size'], self.opts['hidden_size'], self.opts['num_layers'], self.opts['pad_idx'], self.opts['dropout'], self.opts['encoder_attention'], self.opts['attention_time']);

        self.encoder = EncoderRNN(self.opts['vocab_size_encoder'], self.opts['embedding_size'], self.opts['hidden_size'], self.opts['num_layers'], self.opts['pad_idx'], self.opts['dropout'], shared_lt=self.decoder.embedding if self.opts['encoder_shared_lt'] else None)

        optim_class = getattr(optim, optimizer)

        if optimizer == 'Adam':
            self.optimizer = optim_class(self.parameters(), self.opts['lr'], amsgrad=True)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.opts['lr'], nesterov=True, momentum=0.99)
        else:
            self.optimizer = optim_class(self.parameters(), self.opts['lr'])

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.opts['pad_idx'], reduction='sum').to(self.opts['device'])
        
        self.encoder.to(self.opts['device'])
        self.decoder.to(self.opts['device'])

        self.sos_buffer = torch.Tensor([self.opts['sos_idx']]).long().to(self.opts['device'])
        self.longest_label = 40

        self.metrics = {
                'loss': 0.0,
                'num_tokens': 0,
                }


    def reset_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['num_tokens'] = 0
        
    def report_metrics(self):
        if self.metrics['num_tokens'] > 0:
            avg_loss = self.metrics['loss'] / self.metrics['num_tokens']
            ppl = math.exp(avg_loss)
            print('Loss: {}\nPPL: {}'.format(avg_loss, ppl))
            return ppl, avg_loss


    def save_model(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, filename)

    def load_model(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)
        
    def train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()
        
    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_params(self):
        if self.opts['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.opts['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.opts['grad_clip'])
        self.optimizer.step()

    def scheduler_step(self, val_score, min=True):
        if min is False:
            val_score = -val_score
        self.lr_scheduler.step(val_score)

    
    def decode_forced(self, ys, encoder_states, xs_lens):
        encoder_output, encoder_hidden, attention_mask = encoder_states
        
        batch_size = ys.size(0)
        target_length = ys.size(1)
        longest_label = max(target_length, self.longest_label)
        starts = self.sos_buffer.expand(batch_size, 1).long()  # expand to batch size
        
        # Teacher forcing: Feed the target as the next input
        y_in = ys.narrow(1, 0, ys.size(1) - 1)
        decoder_input = torch.cat([starts, y_in], 1)
        decoder_output, decoder_hidden, attn_w_log = self.decoder(decoder_input, encoder_hidden, encoder_states)
        _, preds = decoder_output.max(dim=2)
        
        return decoder_output, preds, attn_w_log
    
    def decode_greedy(self, encoder_states, batch_size):
        encoder_output, encoder_hidden, attention_mask = encoder_states
        
        starts = self.sos_buffer.expand(batch_size, 1)  # expand to batch size
        decoder_hidden = encoder_hidden  # no attention yet

        # greedy decoding here        
        preds = [starts]
        scores = []
        
        finish_mask = torch.Tensor([0]*batch_size).byte().to(self.opts['device'])
        xs = starts
        _attn_w_log = []
        
        for ts in range(self.longest_label):
            decoder_output, decoder_hidden, attn_w_log = self.decoder(xs, decoder_hidden, encoder_states)
            _scores, _preds = F.log_softmax(decoder_output, dim=-1).max(dim=-1)
            preds.append(_preds)
            _attn_w_log.append(attn_w_log)
            scores.append(_scores.view(-1)*(finish_mask == 0).float())

            finish_mask += (_preds == self.opts['eos_idx']).view(-1)
            xs = _preds
            
        return scores, preds, _attn_w_log

    def decode_beam(self, beam_size, batch_size, encoder_states, block_ngram=0, expand_beam=1):
        dev = self.opts['device']
        beams = [ Beam(beam_size, device='cuda', block_ngram=block_ngram, expand_beam=expand_beam) for _ in range(batch_size) ]
        decoder_input = self.sos_buffer.expand(batch_size * beam_size, 1).to(dev)
        inds = torch.arange(batch_size).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        
        encoder_states = self.reorder_encoder_states(encoder_states, inds)  # not reordering but expanding
        incr_state = encoder_states[1]
        
        for ts in range(self.longest_label):
            if all((b.done() for b in beams)):
                break
            score, incr_state, attn_w_log = self.decoder(decoder_input, incr_state, encoder_states)
            score = score[:, -1:, :]
            score = score.view(batch_size, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            
            for i, b in enumerate(beams):
                if not b.done():
                    b.advance(score[i])
                    
            incr_state_inds = torch.cat([beam_size * i + b.get_backtrack_from_current_step() for i, b in enumerate(beams)])
            incr_state = self.reorder_decoder_incremental_state(incr_state, incr_state_inds)
            selection = torch.cat([b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            decoder_input = selection
            
        for b in beams:
            b.check_finished()

        beam_preds_scores = [list(b.get_top_hyp()) for b in beams]
        for pair in beam_preds_scores:
            pair[0] = Beam.get_pretty_hypothesis(pair[0])

        return beam_preds_scores, beams

    def compute_loss(self, encoder_states, xs_lens, ys):
        decoder_output, preds, attn_w_log = self.decode_forced(ys, encoder_states, xs_lens)
        scores = decoder_output.view(-1, decoder_output.size(-1))
        loss = self.criterion(scores, ys.view(-1))
        # normalize loss per non_null num of tokens
        num_tokens = ys.ne(self.opts['pad_idx']).long().sum().item()
        # accumulate metrics
        self.metrics['loss'] += loss.item()
        self.metrics['num_tokens'] += num_tokens
        loss /= num_tokens
        
        return loss
    

    def train_step(self, batch):
        xs, ys, use_packed = batch.text_vecs, batch.label_vecs, batch.use_packed
        xs_lens, ys_lens = batch.text_lens, batch.label_lens

        if xs is None:
            return
        bsz = xs.size(0)
        
        starts = self.sos_buffer.expand(bsz, 1)  # expand to batch size
        self.zero_grad()
        self.train_mode()

        encoder_states = self.encoder(xs, xs_lens, use_packed=use_packed)

        loss = self.compute_loss(encoder_states, xs_lens, ys)
        # pdb.set_trace()
        loss.backward()
        self.update_params()

    def reorder_encoder_states(self, encoder_states, indices):
        """Reorder encoder states according to a new set of indices."""
        enc_out, hidden, attention_mask = encoder_states

        # LSTM or GRU/RNN hidden state?
        if isinstance(hidden, torch.Tensor):
            hid, cell = hidden, None
        else:
            hid, cell = hidden

        if not torch.is_tensor(indices):
            # cast indices to a tensor if needed
            indices = torch.LongTensor(indices).to(hid.device)

        hid = hid.index_select(1, indices)
        if cell is None:
            hidden = hid
        else:
            cell = cell.index_select(1, indices)
            hidden = (hid, cell)

        enc_out = enc_out.index_select(0, indices)
        attention_mask = attention_mask.index_select(0, indices)

        return enc_out, hidden, attention_mask
    
    
    def reorder_decoder_incremental_state(self, incremental_state, inds):
        if torch.is_tensor(incremental_state):
            # gru or lstm
            return torch.index_select(incremental_state, 1, inds).contiguous()
        elif isinstance(incremental_state, tuple):
            return tuple(
                self.reorder_decoder_incremental_state(x, inds)
                for x in incremental_state)


    def eval_step(self, batch, decoding_strategy='score', dump=False):
        xs, ys, use_packed = batch.text_vecs, batch.label_vecs, batch.use_packed
        xs_lens, ys_lens = batch.text_lens, batch.label_lens
            
        self.eval_mode()
        encoder_states = self.encoder(xs, xs_lens, use_packed=use_packed)

        if decoding_strategy == 'score':
            assert ys is not None
            _ = self.compute_loss(encoder_states, xs_lens, ys)
            
        if decoding_strategy == 'greedy':
            scores, preds, attn_w_log = self.decode_greedy(encoder_states, batch.text_vecs.size(0))
            preds = torch.stack(preds, dim=1)
            scores = torch.stack(scores, dim=1)
            pred_lengths = (scores < 0).sum(dim=1).to(scores.device)
            length_penalties = torch.Tensor([Beam.get_length_penalty(i) for i in pred_lengths.tolist()]).to(scores.device)
            scores_length_penalized = scores.sum(dim=1) / length_penalties
            pred_scores = tuple((p, s) for p,s in zip(preds,scores_length_penalized))
            if dump is True:
                _dump = [attn_w_log]
                return pred_scores, _dump
            else:
                return pred_scores

        if 'beam' in decoding_strategy:
            beam_size = int(decoding_strategy.split(':')[1])
            block_ngram = int(decoding_strategy.split(':')[2])
            expand_beam = int(decoding_strategy.split(':')[3])
            beam_pred_scores, beams =self.decode_beam(beam_size, len(batch.text_lens), encoder_states, block_ngram=block_ngram, expand_beam=expand_beam)
            return beam_pred_scores, beams
