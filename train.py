# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import ASGD
import math
import time
import argparse  # Import argparse module
from data import Corpus, batchify
from model import AWDLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parsing
parser = argparse.ArgumentParser(description='AWD-LSTM Language Model Training')

parser.add_argument('--train', action='store_true', help='Train a new model')
parser.add_argument('--load', type=str, help='Path to a saved model to load')
parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on the test set')
parser.add_argument("--bptt", type=int, default=70)
args = parser.parse_args()

# Hyperparameters
batch_size = 20
eval_batch_size = 10
seq_len = 70  # Increased sequence length as per original implementation
epochs = 500  # Set a high number; we'll use early stopping
lr = 30  # Initial learning rate
#lr = 1.17e-01
clip = 0.25  # Gradient clipping
weight_decay = 1.2e-6  # Weight decay (L2 regularization)
alpha = 2.0  # Weight for activation regularization (AR)
beta = 1.0   # Weight for temporal activation regularization (TAR)
nonmono = 0  # Number of epochs to wait before switching to ASGD

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


def evaluate(data_source):
    """Evaluate the model on the validation or test set."""
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / len(data_source)

import numpy as np  # Import numpy for generating random numbers

def get_batch(source, i, seq_len):
    """Get a batch of data from the source with a variable sequence length."""
    seq_len = min(len(source) - 1 - i, seq_len)  # Ensure seq_len doesn't exceed the dataset
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train():
    """Train the model using SGD and switch to ASGD when appropriate."""
    # Set up the optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )

    best_val_loss = None
    stored_loss = float('inf')
    epoch_since_best = 0
    optimizer_type = 'SGD'

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        start_time = time.time()
        hidden = model.init_hidden(batch_size)

        batch, i = 0, 0  # Initialize batch counter and index
        while i < train_data.size(0) - 1 - 1:
            # Dynamic sequence length adjustment
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.  # 95% chance to use args.bptt, 5% chance to use half
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))  # Normal distribution around `bptt`

            # Adjust learning rate based on the sequence length
            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            #print(seq_len, optimizer.param_groups[0]['lr'])
            # Get the batch with the dynamically calculated sequence length
            data, targets = get_batch(train_data, i, seq_len)

            optimizer.zero_grad()
            hidden = repackage_hidden(hidden)

            # Forward pass
            (output, hidden), raw_outputs, outputs = model(data, hidden, return_h=True)
            loss = criterion(output.view(-1, ntokens), targets)

            # Activation Regularization (AR)
            if alpha:
                ar_loss = sum((raw_output**2).mean() for raw_output in raw_outputs)
                loss = loss + alpha * ar_loss

            # Temporal Activation Regularization (TAR)
            if beta:
                tar_loss = sum((h[1:] - h[:-1]).pow(2).mean() for h in raw_outputs)
                loss = loss + beta * tar_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()

            # Reset learning rate back to original value after the step
            optimizer.param_groups[0]['lr'] = lr2

            if batch % 200 == 0 and batch > 0:
                cur_loss = total_loss / 200
                elapsed = time.time() - start_time
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_data) // seq_len:5d} batches | '
                      f'lr {optimizer.param_groups[0]["lr"]:.2e} | ms/batch {elapsed * 1000 / 200:.2f} | '
                      f'loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f}')
                total_loss = 0
                start_time = time.time()

            i += seq_len  # Move forward by the dynamic sequence length
            batch += 1  # Increment the batch counter

        # Evaluate on validation data
        val_loss = evaluate(val_data)
        # After evaluating on validation data
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:.6f} | '
              f'valid ppl {math.exp(val_loss):.2f}')
        print('-' * 89)

        # Print best_val_loss for comparison
        print(f'Current val_loss: {val_loss:.6f}, Best val_loss: {best_val_loss if best_val_loss else "N/A"}')

        # Save the model if the validation loss is the best we've seen so far
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_val_loss = val_loss
            epoch_since_best = 0
        else:
            epoch_since_best += 1

        # Switch to ASGD if no improvement for 'nonmono' epochs
        print("epochs_since_best: " + str(epoch_since_best))
        print(optimizer_type)
        if epoch_since_best >= nonmono and optimizer_type == 'SGD':
            print('=' * 89)
            print('Switching to ASGD')
            print('=' * 89)
            optimizer = ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=1
            )
            optimizer_type = 'ASGD'

        # Adjust learning rate if using SGD
        if optimizer_type == 'SGD':
            scheduler.step(val_loss)
        if optimizer_type == 'ASGD':
            scheduler.step(val_loss)

        # Stop training if validation perplexity is below the target
        if math.exp(val_loss) <= 60:
            print('Validation perplexity below 60, stopping training.')
            break


if args.load:
    print(f'Loading model from {args.load}')
    model.load_state_dict(torch.load(args.load, map_location=device))

if args.train:
    # Call the train function to start training
    train()

if args.evaluate:
    # Evaluate the model on the test set
    test_loss = evaluate(test_data)
    print('=' * 89)
    print(f'| Evaluation | test loss {test_loss:.2f} | test ppl {math.exp(test_loss):.2f}')
    print('=' * 89)
