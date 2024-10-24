import torch
import torch.nn as nn
import numpy as np
import math
import argparse
from torch.autograd import Variable
from data import Corpus, batchify
from model import AWDLSTM
from train import get_batch, repackage_hidden  # Assuming you have these utility functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parsing
parser = argparse.ArgumentParser(description='AWD-LSTM Pointer-based Evaluation')
parser.add_argument('--data', type=str, default='data/penn', help='Location of the data corpus')
parser.add_argument('--load', type=str, help='Path to the saved model to load')
parser.add_argument('--save', type=str, default='best_model.pt', help='Path to save the model')
parser.add_argument('--evaluate', action='store_true', help='Evaluate using the pointer mechanism')
parser.add_argument("--bptt", type=int, default=70, help='Sequence length')
parser.add_argument('--window', type=int, default=3785, help='Pointer window length')
parser.add_argument('--theta', type=float, default=0.6625523432485668, help='Pointer mechanism theta value')
parser.add_argument('--lambdasm', type=float, default=0.12785920428335693,
                    help='Mix between pointer and vocab distribution')
args = parser.parse_args()

# Load data
corpus = Corpus(args.data)
val_data = batchify(corpus.valid, 1, device)
test_data = batchify(corpus.test, 1, device)
ntokens = len(corpus.dictionary)

# Initialize the model
model = AWDLSTM(
    ntokens,
    emb_size=400,
    n_hid=1150,
    n_layers=3,
    dropout=0.4,  # Output dropout
    dropouth=0.25,  # Hidden-to-hidden dropout
    dropouti=0.4,  # Input (embedding) dropout
    dropoute=0.1,
    wdrop=0.5,
    tie_weights=True
).to(device)

criterion = nn.CrossEntropyLoss()


def one_hot(idx, size, cuda=True):
    """One-hot encode the target."""
    a = np.zeros((1, size), np.float32)
    a[0][idx] = 1
    v = Variable(torch.from_numpy(a))
    if cuda:
        v = v.cuda()
    return v


def evaluate(data_source, bptt, window, theta, lambdasm):
    """Evaluate the model using pointer mechanism."""
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(1)  # Assuming batch size of 1 for evaluation
    ntokens = len(corpus.dictionary)
    next_word_history = None
    pointer_history = None

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            hidden = repackage_hidden(hidden)
            (output, hidden), rnn_outs, _ = model(data, hidden, return_h=True)
            rnn_out = rnn_outs[-1].squeeze()
            output_flat = output.view(-1, ntokens)

            # Fill pointer history
            start_idx = len(next_word_history) if next_word_history is not None else 0
            next_word_history = torch.cat([one_hot(t.item(), ntokens) for t in targets]) if next_word_history is None else torch.cat([next_word_history, torch.cat([one_hot(t.item(), ntokens) for t in targets])])

            pointer_history = Variable(rnn_out.data) if pointer_history is None else torch.cat(
                [pointer_history, Variable(rnn_out.data)], dim=0)

            # Manual cross entropy with pointer mechanism
            loss = 0
            softmax_output_flat = torch.nn.functional.softmax(output_flat, dim=-1)
            for idx, vocab_loss in enumerate(softmax_output_flat):
                p = vocab_loss
                if start_idx + idx > window:
                    valid_next_word = next_word_history[start_idx + idx - window:start_idx + idx]
                    valid_pointer_history = pointer_history[start_idx + idx - window:start_idx + idx]
                    logits = torch.mv(valid_pointer_history, rnn_out[idx])
                    ptr_attn = torch.nn.functional.softmax(theta * logits).view(-1, 1)
                    ptr_dist = (ptr_attn.expand_as(valid_next_word) * valid_next_word).sum(0).squeeze()
                    p = lambdasm * ptr_dist + (1 - lambdasm) * vocab_loss

                target_loss = p[targets[idx].data]
                loss += (-torch.log(target_loss)).item()

            total_loss += loss / 1  # Batch size of 1
            next_word_history = next_word_history[-window:]
            pointer_history = pointer_history[-window:]

    return total_loss / len(data_source)


def main():
    # Load the model
    print(f'Loading model from {args.load}')
    model.load_state_dict(torch.load(args.load, map_location=device))


    val_loss = evaluate(val_data, args.bptt, args.window, args.theta, args.lambdasm)
    print('=' * 89)
    print(f'| Pointer-based Evaluation | val loss {val_loss:.2f} | val ppl {math.exp(val_loss):.2f}')
    print('=' * 89)

    test_loss = evaluate(test_data, args.bptt, args.window, args.theta, args.lambdasm)
    print('=' * 89)
    print(f'| Pointer-based Evaluation | test loss {test_loss:.2f} | test ppl {math.exp(test_loss):.2f}')
    print('=' * 89)


# Ensure the script only runs when executed directly
if __name__ == "__main__":
    main()
