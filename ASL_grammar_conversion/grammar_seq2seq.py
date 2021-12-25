from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

inputs_file = Path("ASL_grammar_conversion/x_input_data.txt")
glove = Path("ASL_grammar_conversion/english_glove.txt")
SOS, EOS = 0, 1

with open(inputs_file, mode="r") as inputs:
    raw_inputs = inputs.readlines()
    
def make_embeddings(embeddings_file):
    # uses pretrained embeddings and returns them in a dictionary
    embeddings_index = {}
    with open(embeddings_file, mode="r") as embeddings:
        embeddings = embeddings.readlines()

    # embeddings_index["SOS"] = SOS 
    for line in embeddings:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    # embeddings_index["EOS"] = EOS

    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index


class Vocabulary:
    # Since this is a task where all neccesary outputs will be in the input, maybe a vocabulary could be useful?
    def __init__(self, title, embeddings, non_ASL_words):
        # Takes in a sentence and makes it into a decision space for the Seq2Seq
        self.title = title
        self.non_ASL_words = non_ASL_words
        self.vocab = {word:[None, embeddings[word]] for word in non_ASL_words}
        
        self.n_words = len(non_ASL_words) # + 2 for SOS and EOS
        # list of words for the Seq2Seq to pick from
        self.words = [word for word in non_ASL_words]
        
        self.embeddings_dict = embeddings
        
    def __repr__(self):
        # enables print(vocabulary)
        return str(self.vocab)
    
    def word2vocab(self, word):
        word = word.lower()
        self.words.append(word)
        self.n_words = len(list(self.vocab.keys()))
        self.vocab[word] = [self.words.count(word), self.embeddings_dict[word]] # word: [count, embedding]
    
    def remove_word(self, word):
        word = word.lower()
        
        if word not in self.non_ASL_words:
            n = self.vocab[word][0] - 1 # removes one from the count of that word
            if n == 0:
                del self.vocab[word]
            else:
                self.vocab[word][0] = n
        
    def reset_vocab(self):
        self.vocab = {}
        self.words = [word for word in self.non_ASL_words]
        self.n_words = len(self.non_ASL_words)

 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size=50, context_size=50):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.context_size = context_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, ht, ct):
        output = embedded
        output, (ht, ct) = self.lstm(output, (ht, ct))

        return output, (ht, ct)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def init_context(self):
        return torch.zeros(1, 1, self.context_size)

    def get_params(self):
        pass

class DecoderRNN:
    def __init__(self, output_size, hidden_size=50, context_size = 50):
        # output size is sentence length
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, output, ct, ht):
        output = torch.Tensor(output).view(1, 1, -1)

        output, (ht, ct) = self.lstm(output)
        output = F.relu(self.out(output))
        output = self.softmax(output).view(1, -1)

        return output, (ct, ht)
        
        
embeddings = make_embeddings(embeddings_file=glove)
embeddings_length = len(embeddings)
sent = "hello"
sent_embeddings = [embeddings[word] for word in sent.split()]

encoder = EncoderRNN(50)
decoder = DecoderRNN(embeddings_length)

ht, ct = encoder.init_hidden(), encoder.init_context()
words = list(embeddings.values())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ENCODER~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for i in range(len(sent_embeddings)):
    output, (ht, ct) = encoder.forward(sent_embeddings[i], ht, ct)

    print(f"Output: {output}")
    print(f"Hidden: {ht}")
    print(f"Context: {ct}\n\n")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DECODER~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
m = nn.Linear(50, embeddings_length)
output = m(output)
print(type(output))
for i in range(2):
    # should have it stop outputting when the EOS is returned
    choice = torch.argmax(output)
    output = torch.Tensor(embeddings[list(embeddings)[choice]])
    print(list(embeddings)[choice])
    output, (ht, ct) = decoder.forward(output, ht, ct)
    print(f"Output: {output}")
    print(f"Hidden: {ht}")
    print(f"Context: {ct}\n\n")
    
