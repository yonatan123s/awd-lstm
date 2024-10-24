import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import ASGD
import math
import time
import argparse  # Import argparse module
from data import Corpus, batchify
from model import AWDLSTM
import numpy as np  # Import numpy for generating random numbers
import torch.nn.functional as F  # Import for softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 20
eval_batch_size = 10
seq_len = 70  # Increased sequence length as per original implementation
epochs = 500  # Set a high number; we'll use early stopping
lr = 30  # Initial learning rate
clip = 0.25  # Gradient clipping
weight_decay = 1.2e-6  # Weight decay (L2 regularization)
alpha = 2.0  # Weight for activation regularization (AR)
beta = 1.0   # Weight for temporal activation regularization (TAR)
nonmono = 5  # Number of epochs to wait before switching to ASGD

# Load data
corpus = Corpus('ptbdataset')
train_data = batchify(corpus.train, batch_size, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)
ntokens = len(corpus.dictionary)

# Initialize the model
model = AWDLSTM(
    ntokens,
    emb_size=400,
    n_hid=1150,
    n_layers=3,
    dropout=0.4,     # Output dropout
    dropouth=0.25,   # Hidden-to-hidden dropout
    dropouti=0.4,    # Input (embedding) dropout
    dropoute=0.1,
    wdrop=0.5,
    tie_weights=True
).to(device)

criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return [repackage_hidden(v) for v in h]

def evaluate(data_source, bptt):
    """Evaluate the model on the validation or test set."""
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / len(data_source)

def get_batch(source, i, seq_len):
    """Get a batch of data from the source with a variable sequence length."""
    seq_len = min(len(source) - 1 - i, seq_len)  # Ensure seq_len doesn't exceed the dataset
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train(bptt):
    """Train the model using SGD and switch to ASGD when appropriate."""
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    best_val_loss = None
    stored_loss = float('inf')
    epoch_since_best = 0
    optimizer_type = 'SGD'

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        start_time = time.time()
        hidden = model.init_hidden(batch_size)

        batch, i = 0, 0
        while i < train_data.size(0) - 1 - 1:
            bptt_seq_len = bptt if np.random.random() < 0.95 else bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt_seq_len, 5)))
            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / bptt
            data, targets = get_batch(train_data, i, seq_len)

            optimizer.zero_grad()
            hidden = repackage_hidden(hidden)
            (output, hidden), raw_outputs, outputs = model(data, hidden, return_h=True)
            loss = criterion(output.view(-1, ntokens), targets)

            if alpha:
                ar_loss = sum((raw_output**2).mean() for raw_output in raw_outputs)
                loss += alpha * ar_loss

            if beta:
                tar_loss = sum((h[1:] - h[:-1]).pow(2).mean() for h in raw_outputs)
                loss += beta * tar_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            optimizer.param_groups[0]['lr'] = lr2

            if batch % 200 == 0 and batch > 0:
                cur_loss = total_loss / 200
                elapsed = time.time() - start_time
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_data) // seq_len:5d} batches | '
                      f'lr {optimizer.param_groups[0]["lr"]:.2e} | ms/batch {elapsed * 1000 / 200:.2f} | '
                      f'loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f}')
                total_loss = 0
                start_time = time.time()

            i += seq_len
            batch += 1

        val_loss = evaluate(val_data, bptt)
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:.6f} | valid ppl {math.exp(val_loss):.2f}')
        print('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_val_loss = val_loss
            epoch_since_best = 0
        else:
            epoch_since_best += 1

        if epoch_since_best >= nonmono and optimizer_type == 'SGD':
            print('=' * 89)
            print('Switching to ASGD')
            optimizer = ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
            optimizer_type = 'ASGD'

        if optimizer_type == 'SGD':
            scheduler.step(val_loss)

        if math.exp(val_loss) <= 60:
            print('Validation perplexity below 60, stopping training.')
            break


def predict_top_5(sequence):
    """Given a sequence, predict the top 5 next words and their probabilities."""
    model.eval()
    tokens = []
    unk_idx = corpus.dictionary.word2idx.get('<unk>')  # Get the index for '<unk>'
    for word in sequence.split():
        idx = corpus.dictionary.word2idx.get(word, unk_idx)
        tokens.append(idx)
    input_tensor = torch.tensor(tokens).unsqueeze(1).to(device)  # Shape: (seq_len, 1)
    hidden = model.init_hidden(1)

    with torch.no_grad():
        output, hidden = model(input_tensor, hidden)
        output = output[-1, 0, :]  # Get the last output vector for batch element 0
        probabilities = F.softmax(output, dim=0)  # Apply softmax on the output

    top_5_prob, top_5_idx = torch.topk(probabilities, 5)
    top_5_words = [corpus.dictionary.idx2word[idx.item()] for idx in top_5_idx]

    return list(zip(top_5_words, top_5_prob.tolist()))



def interactive_mode():
    """Interactive mode for user to input sequences and see top 5 predictions."""
    print("Enter a sequence to predict the next word (type 'terminate run' to exit):")
    while True:
        sequence = input("Input sequence: ").strip()
        if sequence.lower() == "terminate run":
            break

        predictions = predict_top_5(sequence)
        if predictions:
            print("Top 5 predictions and probabilities:")
            for word, prob in predictions:
                print(f"{word}: {prob:.4f}")
        else:
            print("No predictions available for the given sequence.")

# Main function for argument parsing and execution
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='AWD-LSTM Language Model Training')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--load', type=str, help='Path to a saved model to load')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on the test set')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode to input sequences')
    parser.add_argument("--bptt", type=int, default=70)
    args = parser.parse_args()

    if args.load:
        print(f'Loading model from {args.load}')
        model.load_state_dict(torch.load(args.load, map_location=device))

    if args.train:
        train(args.bptt)

    if args.evaluate:
        test_loss = evaluate(test_data, args.bptt)
        print('=' * 89)
        print(f'| Evaluation | test loss {test_loss:.2f} | test ppl {math.exp(test_loss):.2f}')
        print('=' * 89)

    if args.interactive:
        interactive_mode()

# Ensure the script only runs when executed directly
if __name__ == "__main__":
    main()
