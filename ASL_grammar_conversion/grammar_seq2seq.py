from pathlib import Path
import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ASL_grammar_data_gen import english_to_ASL

inputs_file = Path("x_input_data.txt")
GloVe_file = "ASL_grammar_conversion/glove/glove.6B.50d.txt"

with open(inputs_file, mode="r") as inputs:
    raw_inputs = inputs.readlines()


def make_embeddings(embeddings_file):
    # uses pretrained embeddings and returns them in a dictionary
    embeddings_index = {}
    with open(embeddings_file, mode="r") as embeddings:
        embeddings = embeddings.readlines()

    for line in embeddings:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index


"""
Should take in glove embedding vector for one word, a hidden vector (first one is zeros) and a context vector (again made of zeros) then 
output a hidden vector, a represenation of all the words before the previous token (context vector),
and an output that is the last word in the right grammar order
context vector is used to keep a running track of the info, 
hidden state keeps track of what information to combine into the context vector from the input it gets combined with
hence: (inputs, (ht, ct))
"""


class EncoderRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size=50, context_size=50, pretrained_embeddings=None
    ):
        # pretrained embeddings should be a dictionary
        super(EncoderRNN, self).__init__()

        self.pretrained_embeddings = pretrained_embeddings
        self.hidden_size = hidden_size
        self.context_size = context_size

        if pretrained_embeddings:
            self.embedding = pretrained_embeddings
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)

        self.lstm = nn.LSTM(input_size, hidden_size)
        print(f"lstm: {type(self.lstm)}")

    def forward(self, input, ht, ct):
        # completes one step of the forward pass for the encoder of the sequence to sequence model.
        # input should be the next word and it will be converted to an embedding.

        if self.pretrained_embeddings:
            embedded = torch.Tensor(self.embedding[input]).view(1, 1, -1)
        else:
            embedded = self.embedding(input).view(1, 1, -1)

        output = embedded
        output, (ht, ct) = self.lstm(output, (ht, ct))

        return output, (ht, ct)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def init_context(self):
        return torch.zeros(1, 1, self.context_size)

    # dont forget to append an end token

embeddings_idx = make_embeddings(embeddings_file=GloVe_file)
embeddings_length = np.shape(list(embeddings_idx.values()))[1]

encoder = EncoderRNN(embeddings_length, pretrained_embeddings=embeddings_idx)

output, (ht, ct) = encoder.forward(
    "word", ht=encoder.init_hidden(), ct=encoder.init_context()
)

print(f"output: {output}")
