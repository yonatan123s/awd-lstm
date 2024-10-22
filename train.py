# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from data import Corpus, batchify
from model import AWDLSTM
from torch.optim import ASGD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 80
eval_batch_size = 10
seq_len = 35
epochs = 40
lr = 30  # Initial learning rate
clip = 0.25  # Gradient clipping
best_val_loss = None
nonmono = 5  # Number of epochs to wait before switching to ASGD
epoch_since_best = 0
optimizer_type = 'SGD'  # Start with SGD

# Load data
corpus = Corpus('ptbdataset')

train_data = batchify(corpus.train, batch_size, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)

ntokens = len(corpus.dictionary)
model = AWDLSTM(
    ntokens,
    emb_size=400,
    n_hid=1150,
    n_layers=3,
    dropout=0.5,
    dropouth=0.4,
    dropouti=0.65,
    wdrop=0.5,
    tie_weights=True  # Enable weight tying
).to(device)

criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return [repackage_hidden(v) for v in h]


def get_batch(source, i):
    seq_len = min(len(source) - 1 - i, 35)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)

# Hyperparameters for AR and TAR
alpha = 2.0  # Weight for activation regularization
beta = 1.0   # Weight for temporal activation regularization

# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import ASGD
import math
import time
from data import Corpus, batchify
from model import AWDLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 80
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
    dropouti=0.65,   # Input (embedding) dropout
    wdrop=0.5,
    tie_weights=True
).to(device)

criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(len(source) - 1 - i, 70)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def evaluate(data_source):
    """Evaluate the model on the validation or test set."""
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data, targets = get_batch(data_source, i)
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / len(data_source)

def train():
    """Train the model using SGD and switch to ASGD when appropriate."""
    # Set up the optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.25, patience=0
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

        for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            hidden = repackage_hidden(hidden)

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

            if batch % 200 == 0 and batch > 0:
                cur_loss = total_loss / 200
                elapsed = time.time() - start_time
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_data) // seq_len:5d} batches | '
                      f'lr {optimizer.param_groups[0]["lr"]:.2e} | ms/batch {elapsed * 1000 / 200:.2f} | '
                      f'loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f}')
                total_loss = 0
                start_time = time.time()

        # Evaluate on validation data
        val_loss = evaluate(val_data)
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:.2f} | '
              f'valid ppl {math.exp(val_loss):.2f}')
        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_val_loss = val_loss
            epoch_since_best = 0
        else:
            epoch_since_best += 1

        # Switch to ASGD if no improvement for 'nonmono' epochs
        if epoch_since_best >= nonmono and optimizer_type == 'SGD':
            print('=' * 89)
            print('Switching to ASGD')
            print('=' * 89)
            optimizer = ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=weight_decay)
            optimizer_type = 'ASGD'

        # Adjust learning rate if using SGD
        if optimizer_type == 'SGD':
            scheduler.step(val_loss)

        # Stop training if validation perplexity is below the target
        if math.exp(val_loss) <= 60:
            print('Validation perplexity below 60, stopping training.')
            break

    # Load the best saved model
    model.load_state_dict(torch.load('best_model.pt'))

    # Run on test data
    test_loss = evaluate(test_data)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:.2f} | test ppl {math.exp(test_loss):.2f}')
    print('=' * 89)



# Call the train function to start training
train()


# Load the best saved model.
model.load_state_dict(torch.load('best_model.pt'))

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
print('=' * 89)
