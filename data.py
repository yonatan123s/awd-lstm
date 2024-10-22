# data.py

import os
import torch

class Dictionary:
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.counter = {'<unk>': 0}  # Initialize the counter for '<unk>'

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
            self.counter[word] = 1
        else:
            self.counter[word] += 1

    def __len__(self):
        return len(self.idx2word)

class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Build the vocabulary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx.get(word, self.dictionary.word2idx['<unk>'])
                    token += 1
        return ids

def batchify(data, batch_size, device):
    """Divides the data into batches."""
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)
