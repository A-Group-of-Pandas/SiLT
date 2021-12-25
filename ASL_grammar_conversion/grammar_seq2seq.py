from pathlib import Path
# import typing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# avenuse of improvement: glove has multiple languages, increase glove size, change network arch or params, context and hidden sizes
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
        # there is None here because the non_asl_words should have the same structure as the rest,
        # but they can be used as many times as the Seq2Seq predicts them to be
        self.vocab = {word:[None, embeddings[word]] for word in non_ASL_words}
        
        self.n_words = len(non_ASL_words) # + 2 for SOS and EOS
        # list of words for the Seq2Seq to pick from
        self.words = [word for word in non_ASL_words]
        
        self.embeddings_dict = embeddings
        
    def __repr__(self):
        # enables print(vocabulary)
        return str(self.vocab)
    
    def word2vocab(self, word):
        # could have a problem where a word isnt in the embeddings
        word = word.lower()
        self.words.append(word)
        self.n_words = len(list(self.vocab.keys()))
        self.vocab[word] = [self.words.count(word), self.embeddings_dict[word]] # word: [count, embedding]
    
    def remove_word(self, word):
        word = word.lower()
        
        # returning this is silly bc the words will be chosen from the words list which only contains words in the vocab
        # if word not in list(self.vocab.keys()):  #this lookup is slow
            # return"That is not in the vocabulary"
            # make a gaydar
            
        # removes the word as its chosen from the vocab so it isnt selected twice unlsess its a non-asl word
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


# for word in sentence append word:embedding to vocab with and or, the, of, is in front and make those the decision space
 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size=50, context_size=50):
        # pretrained embeddings should be a dictionary
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.context_size = context_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, ht, ct):
        # input should be a embedding
        # completes one step of the forward pass for the encoder of the sequence to sequence model.
        # input should be the next word and it will be converted to an embedding.
        embedded = torch.Tensor(input).view(1, 1, -1)
        output = embedded
        output, (ht, ct) = self.lstm(output, (ht, ct))

        return output, (ht, ct)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def init_context(self):
        return torch.zeros(1, 1, self.context_size)

    def get_params(self):
        pass

    # dont forget to append an end token
# put encoder output through a final hidden layer (changing dimensionality to 10,000 (one prediction for every word in glove)) and then through a softmax layer and then take the word predicted by the output and get the glove emedding for it as the hidden  
class DecoderRNN:
    #make output, then take that word embeddoing and pt it as input for next one
    def __init__(self, output_size, hidden_size=50, context_size = 50):
        # output size is sentence length
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        # test w/o the linear layer also
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, output, ct, ht):
        # take in output of previous layer (which should be a word vector) with a hidden layer (representation of the sentence up to that point) 
        # with context vector (a helper vector for figuring out what it needs to remember) and make a new categorical choice of word from the dictionary/vocabulary
        # the vocab could either be made of the sentence and any stop/preposition words not in ASL OR it could be learned over time using the target vectors
        # output, (ct, ht) = self.lstm(output, (ht, ct)
        # )
        
        #  change 'output' to be labeld as smthg else
        output = torch.Tensor(output).view(1, 1, -1)

        output, (ht, ct) = self.lstm(output)
        output = F.relu(self.out(output))
        output = self.softmax(output).view(1, -1)

        return output, (ct, ht)
        
        
embeddings = make_embeddings(embeddings_file=glove)
embeddings_length = len(embeddings)

# test_vocabulary = Vocabulary('test', embedding_bag, non_ASL)

# we pass the encoder a EOS token bc some of the intermediate states migth be intentioned (by the encoder) to be not very meagningful so that
# the next step is, passing an EOS tells it that we will judge loss on the next thing it spits out

# the EOS is useful for the decoder bc it can tell us when its done translating
sent = "hello"
sent_embeddings = [embeddings[word] for word in sent.split()]
# sent_embeddings.insert(0, torch.Tensor(np.zeros(50)).view(1, 1, -1))
# sent_embeddings.append(1, torch.Tensor(np.zeros(50)).view(1, 1, -1))
# for word in sent.split():
#     test_vocabulary.word2vocab(word)

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
    
