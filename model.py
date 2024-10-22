# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math

class WeightDropLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, weight_dropout=0.5):
        super(WeightDropLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_dropout = weight_dropout

        # Define weights and biases
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hidden):
        hx, cx = hidden
        # Apply weight dropout to hidden weights
        if self.training:
            w_hh = F.dropout(self.weight_hh, p=self.weight_dropout, training=True)
        else:
            w_hh = self.weight_hh

        gates = (torch.mm(input, self.weight_ih.t()) +
                 torch.mm(hx, w_hh.t()) + self.bias)

        # Split the gates into their respective components
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy
class LockedDropout(nn.Module):
    """Implements locked (variational) dropout."""
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - dropout)
        mask = mask.div_(1 - dropout)
        mask = mask.expand_as(x)
        return x * mask

class EmbeddingDropout(nn.Module):
    """Applies dropout to the embedding layer by zeroing out entire words."""
    def forward(self, embed, words, dropout=0.1):
        if dropout and self.training:
            mask = embed.weight.new_empty((embed.weight.size(0), 1), requires_grad=False).bernoulli_(1 - dropout)
            mask = mask.expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        padding_idx = embed.padding_idx if embed.padding_idx is not None else -1
        return F.embedding(
            words,
            masked_embed_weight,
            padding_idx,
            embed.max_norm,
            embed.norm_type,
            embed.scale_grad_by_freq,
            embed.sparse)

class WeightDrop(nn.Module):
    """A wrapper around RNN modules that applies weight dropout."""
    def __init__(self, module, weights, dropout):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        # Remove weights from the module and store them as parameters in this wrapper
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            # Remove the parameter from the module
            del self.module._parameters[name_w]
            # Register it as a parameter in this wrapper
            self.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        # Apply dropout to the raw weights and assign them back to the module
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            # Apply dropout during training
            if self.training:
                w = F.dropout(raw_w, p=self.dropout, training=True)
            else:
                w = raw_w
            # Assign the modified weight to the module
            setattr(self.module, name_w, w)

    def forward(self, *args, **kwargs):
        self._setweights()
        with torch.backends.cudnn.flags(enabled=False):
            return self.module(*args, **kwargs)

class AWDLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size=400, n_hid=1150, n_layers=3,
                 dropout=0.4, dropouth=0.25, dropouti=0.65, wdrop=0.5,
                 tie_weights=True):
        super().__init__()
        self.lockdrop = LockedDropout()
        self.emb_dropout = EmbeddingDropout()
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.n_layers = n_layers
        self.hidden_sizes = []
        self.rnns = []

        for l in range(n_layers):
            input_size = emb_size if l == 0 else n_hid
            hidden_size = n_hid if l != n_layers - 1 else (emb_size if tie_weights else n_hid)
            self.hidden_sizes.append(hidden_size)
            rnn = WeightDropLSTM(input_size, hidden_size, weight_dropout=wdrop)
            self.rnns.append(rnn)

        self.rnns = nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(emb_size if tie_weights else n_hid, vocab_size)
        self.init_weights()
        self.emb_size = emb_size
        self.n_hid = n_hid
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.tie_weights = tie_weights

        if self.tie_weights:
            # Tie weights
            self.decoder.weight = self.encoder.weight

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, input, hidden, return_h=False):
        emb = self.emb_dropout(self.encoder, input, dropout=self.dropouti)
        emb = self.lockdrop(emb, self.dropouti)
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output = []
            hx, cx = hidden[l]  # Get the initial hidden state for this layer

            for time_step in range(current_input.size(0)):
                input_t = current_input[time_step]
                hy, cy = rnn(input_t, (hx, cx))
                hx, cx = hy, cy  # Update hidden state
                raw_output.append(hy.unsqueeze(0))

            raw_output = torch.cat(raw_output, 0)
            new_hidden.append((hx, cx))  # Save the new hidden state
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1)), new_hidden
        if return_h:
            return result, raw_outputs, outputs
        else:
            return result

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return [(weight.new_zeros(batch_size, self.hidden_sizes[l]),
                 weight.new_zeros(batch_size, self.hidden_sizes[l]))
                for l in range(self.n_layers)]

