import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: unlike official PyTorch tutorial, this model doesn't process one sample at a time
# because it's slow on GPU. Instead it uses masks just like ye olde Tensorflow.
# it doesn't use torch.nn.utils.rnn.pack_paded_sequence because reasons.

# https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training

class Attention(nn.Module):
    def __init__(self, hid_size):
        super().__init__()
        self.hid_size = hid_size
        self.prev_l = nn.Linear(hid_size, hid_size)
        self.encoder_l = nn.Linear(hid_size, hid_size)
        self.logits_l = nn.Linear(hid_size, 1)

    def forward(self, prev_dec, encoder_out):
        # prev_dec: [B, H]
        # encoder_out: [B, T_in, H]
        # prev_dec_reshaped: [B, 1, H]
        # prev_encoder_sum: [B, T_in, H]
        # scores: [B, T_in, H]
        # weights_logits: [B, T_in, 1]
        prev_dec_reshaped = prev_dec.reshape(-1, 1, self.hid_size)
        prev_encoder_sum = self.prev_l(prev_dec_reshaped) + self.encoder_l(encoder_out)
        weights_logits = self.logits_l(torch.tanh(prev_encoder_sum))

        # weights_logits.squeeze(2).unsqueeze(1): [B, 1, T_in]
        # weights: [B, 1, T_in]
        weights = F.softmax(weights_logits.squeeze(2).unsqueeze(1), dim=-1)
        context = torch.bmm(weights, encoder_out)

        # context.squeeze(1): [B, H]
        return context.squeeze(1)


class Encoder(nn.Module):
    def __init__(self, inp_voc, emb_size, hid_size):
        super().__init__()
        self.inp_voc = inp_voc
        self.hid_size = hid_size
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size, padding_idx=self.inp_voc.pad_ix)
        self.bi_gru = nn.GRU(emb_size, hid_size, batch_first=True, bidirectional=True)
        self.dec_start_l = nn.Linear(2 * hid_size, hid_size)
        self.dec_out_l = nn.Linear(2 * hid_size, hid_size)
    
    def forward(self, inp):
        """
        Takes symbolic input sequence, computes initial state for decoder
        """
        end_index = infer_length(inp, self.inp_voc.eos_ix)
        end_index[end_index >= inp.shape[1]] = inp.shape[1] - 1
        inp_emb = self.emb_inp(inp)

        enc_seq, h_final = self.bi_gru(inp_emb)  # enc_seq.shape: [B, T, 2 * H]
        enc_seq_2dir = enc_seq.view(enc_seq.shape[0], enc_seq.shape[1], 2, self.hid_size)  # [B, T, 2, H]
        enc_last_backward = h_final[1, :, :]
        enc_last = enc_seq_2dir[range(0, enc_seq.shape[0]), end_index.detach(), :, :]  # [B, 2,  H]
        enc_last_forward = enc_last[:, 0, :]  # [B, H]
        enc_seq_out = self.dec_out_l(enc_seq)
        
        dec_start = self.dec_start_l(torch.cat((enc_last_forward, enc_last_backward), dim=-1))

        return [[dec_start], enc_seq_out]
        

class DecoderCell(nn.Module):
    def __init__(self, out_voc, emb_size, hid_size):
        super().__init__()
        self.out_voc = out_voc
        self.gru = nn.GRUCell(emb_size + hid_size, hid_size)
        self.attention = Attention(hid_size)
        self.logits_l = nn.Linear(hid_size, len(out_voc))
        self.emb_out = nn.Embedding(len(out_voc), emb_size, padding_idx=self.out_voc.pad_ix)

    def forward(self, prev_state, prev_tokens, encoder_out):
        """
        Takes previous decoder state and tokens, returns new state and logits
        """
        # prev_tokens: [B]
        # encoder_out: [B, T_in, H]
        # prev_dec: [B, H]
        [prev_dec] = prev_state
        prev_tokens_emb = self.emb_out(prev_tokens)
        # prev_tokens_emb: [B, emb_size]
        # context: [B, H]
        context = self.attention(prev_dec, encoder_out)

        new_dec_state = self.gru(torch.cat((prev_tokens_emb, context), dim=-1), prev_dec)
        output_logits = self.logits_l(new_dec_state)

        return [new_dec_state], output_logits


class TranslationModel(nn.Module):
    def __init__(self, inp_voc, out_voc,
                 emb_size, hid_size,):
        super(self.__class__, self).__init__()
        self.inp_voc = inp_voc
        self.out_voc = out_voc

        self.encoder = Encoder(inp_voc, emb_size, hid_size)
        self.decoder = DecoderCell(out_voc, emb_size, hid_size)

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: input tokens, int64 vector of shape [batch]
        :return: a list of initial decoder state tensors
        """
        return self.encoder(inp)

    def decode(self, prev_state, prev_tokens, encoder_out, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """
        return self.decoder(prev_state, prev_tokens, encoder_out)

    def forward(self, inp, out, eps=1e-30, **flags):
        """
        Takes symbolic int32 matrices of hebrew words and their english translations.
        Computes the log-probabilities of all possible english characters given english prefices and hebrew word.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param out: output sequence, int32 matrix of shape [batch,time]
        :return: log-probabilities of all possible english characters of shape [batch,time,n_tokens]

        Note: log-probabilities time axis is synchronized with out
        In other words, logp are probabilities of __current__ output at each tick, not the next one
        therefore you can get likelihood as logprobas * tf.one_hot(out,n_tokens)
        """
        device = next(self.parameters()).device
        inp = inp.to(device)
        out = out.to(device)
        batch_size = inp.shape[0]
        bos = torch.tensor(
            [self.out_voc.bos_ix] * batch_size,
            dtype=torch.long,
            device=device,
        )
        logits_seq = [torch.log(to_one_hot(bos, len(self.out_voc)) + eps)]
        # logits_seq[0] <-- [batch_size, n_tokens]

        hid_state, enc_seq = self.encode(inp, **flags)
        for x_t in out.transpose(0, 1)[:-1]:
            hid_state, logits = self.decode(hid_state, x_t, enc_seq, **flags)
            logits_seq.append(logits)

        # torch.stack(logits_seq, dim=1) <-- [B, T_out, V]
        # V = n tokens
        return F.log_softmax(torch.stack(logits_seq, dim=1), dim=-1)

    def translate(self, inp, greedy=False, max_len=None, eps=1e-30, **flags):
        """
        takes symbolic int32 matrix of hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param greedy: if greedy, takes token with highest probablity at each tick.
            Otherwise samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :return: output tokens int32[batch,time] and
                 log-probabilities of all tokens at each tick, [batch,time,n_tokens]
        """
        device = next(self.parameters()).device
        inp = inp.to(device)
        batch_size = inp.shape[0]
        bos = torch.tensor(
            [self.out_voc.bos_ix] * batch_size,
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones(batch_size, dtype=torch.uint8, device=device)
        

        logits_seq = [torch.log(to_one_hot(bos, len(self.out_voc)) + eps)]
        # we need this layer, because without it
        # next word which this layer predicts will be relatively shifted by 1
        # By adding this layer we will level that shift
        
        out_seq = [bos]

        hid_state, enc_seq = self.encode(inp, **flags)
        while True:  # auto-regressive inference
            hid_state, logits = self.decode(hid_state, out_seq[-1], enc_seq, **flags)
            if greedy:
                _, y_t = torch.max(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                y_t = torch.multinomial(probs, 1)[:, 0]

            logits_seq.append(logits)
            out_seq.append(y_t)
            mask &= y_t != self.out_voc.eos_ix

            if not mask.any():
                break
            if max_len and len(out_seq) >= max_len:
                break

        return (
            torch.stack(out_seq, 1),  # [B, T_gen] 
            F.log_softmax(torch.stack(logits_seq, 1), dim=-1),  # [B, T_gen, V]
        )


### Utility functions ###

def infer_mask(
        seq,
        eos_ix,
        batch_first=True,
        include_eos=True,
        dtype=torch.float):
    """
    compute mask given output indices and eos code
    :param seq: tf matrix [batch,time] if batch_first else [time,batch]
    :param eos_ix: integer index of end-of-sentence token
    :param include_eos: if True, the time-step where eos first occurs is has mask = 1
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    assert seq.dim() == 2
    is_eos = (seq == eos_ix).to(dtype=torch.float)
    if include_eos:
        if batch_first:
            is_eos = torch.cat((is_eos[:, :1] * 0, is_eos[:, :-1]), dim=1)
        else:
            is_eos = torch.cat((is_eos[:1, :] * 0, is_eos[:-1, :]), dim=0)
    count_eos = torch.cumsum(is_eos, dim=1 if batch_first else 0)
    mask = count_eos == 0
    return mask.to(dtype=dtype)


def infer_length(
        seq,
        eos_ix,
        batch_first=True,
        include_eos=True,
        dtype=torch.long):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :param include_eos: if True, the time-step where eos first occurs is has mask = 1
    :returns: lengths, int32 vector of shape [batch]
    """
    mask = infer_mask(seq, eos_ix, batch_first, include_eos, dtype)
    return torch.sum(mask, dim=1 if batch_first else 0)


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.to(dtype=torch.long).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0],
        n_dims,
        device=y.device,
    ).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot
