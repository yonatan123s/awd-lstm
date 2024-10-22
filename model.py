# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class LockedDropout(nn.Module):
    """Implements locked (variational) dropout."""
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = mask.div_(1 - dropout)
        mask = mask.expand_as(x)
        return x * mask

class WeightDropLSTM(nn.LSTM):
    """Applies weight dropout to an LSTM's hidden-to-hidden weights."""
    def __init__(self, *args, weight_dropout=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dropout = weight_dropout
        self._setup()

    def _setup(self):
        w_names = ['weight_hh_l0']
        for name in w_names:
            w = getattr(self, name)
            del self._parameters[name]
            self.register_parameter(name + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        raw_w = getattr(self, 'weight_hh_l0_raw')
        w = F.dropout(raw_w, p=self.weight_dropout, training=self.training)
        setattr(self, 'weight_hh_l0', w)

    def forward(self, *args):
        self._setweights()
        # Disable cuDNN within this context
        with cudnn.flags(enabled=False):
            return super().forward(*args)

class AWDLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size=400, n_hid=1150, n_layers=3,
                 dropout=0.4, dropouth=0.25, dropouti=0.4, wdrop=0.5):
        super().__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.rnns = nn.ModuleList()
        self.hidden_sizes = []

        for l in range(n_layers):
            input_size = emb_size if l == 0 else n_hid
            hidden_size = n_hid if l != n_layers - 1 else emb_size
            self.hidden_sizes.append(hidden_size)
            rnn = WeightDropLSTM(
                input_size,
                hidden_size,
                1, batch_first=False, weight_dropout=wdrop)
            self.rnns.append(rnn)

        self.decoder = nn.Linear(emb_size, vocab_size)
        self.init_weights()
        self.n_layers = n_layers
        self.n_hid = n_hid
        self.emb_size = emb_size
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth


    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, input, hidden, return_h=False):
        emb = self.encoder(input)
        emb = self.lockdrop(emb, self.dropouti)
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
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
        return [(weight.new_zeros(1, batch_size, h_size),
                 weight.new_zeros(1, batch_size, h_size))
                for h_size in self.hidden_sizes]

