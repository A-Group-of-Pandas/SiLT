from pathlib import Path
import numpy as np
import torch
from torch import nn

inputs_file = Path("ASL_grammar_conversion/x_input_data.txt")
glove = Path("ASL_grammar_conversion/english_glove.txt")

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
        embeddings_index[word] = torch.Tensor(coefs)

    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index
 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size=50, context_size=50):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.context_size = context_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, ht, ct):
        embedded = torch.Tensor(input).view(1, 1, -1)
        output = embedded
        output, (ht, ct) = self.lstm(output, (ht, ct))

        return output, (ht, ct)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def init_context(self):
        return torch.zeros(1, 1, self.context_size)


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size=50):
        # output size is sentence length
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, output, ct, ht):
        output = torch.Tensor(output).view(1, 1, -1)

        output, (ht, ct) = self.lstm(output)
        output = self.softmax(self.out(output))

        return output, (ht, ct)
        
