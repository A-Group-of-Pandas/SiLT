from grammar_seq2seq import *
import string
import torch
import random
from torch import optim
import torch.nn as nn
# import tqdm

SOS_TOKEN = torch.zeros(50)
EOS_TOKEN = torch.ones(50)

teacher_forcing_ratio = 0.65


def sentenceToTensor(embeddings, sentence):
    # embeddings is a dict
    sent = sentence.lower().strip()
    sent = sent.translate(str.maketrans("", "", string.punctuation))
    
    tensor = [embeddings[word] if word in embeddings else torch.eye(len(embeddings))[i-1] for i, word in enumerate(sent.split())]
    tensor.append(EOS_TOKEN)
    
    return tensor


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, embeddings):
    # batching?
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = len(input_tensor)
    target_length = len(target_tensor)
    
    loss = 0
        
    enc_hidden, enc_context = encoder.init_hidden(), encoder.init_context()

    for ei in range(input_length):
        enc_output, (enc_hidden, enc_context) = encoder.forward(input_tensor[ei], enc_hidden, enc_context)
    
    dec_input = SOS_TOKEN
    dec_hidden, dec_context = enc_hidden, enc_context
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        for di in range(target_length):
            dec_output, (dec_hidden, dec_context) = decoder.forward(dec_input, dec_hidden, dec_context)
            dec_output = list(embeddings.values())[torch.argmax(dec_output)] # output size is however big the glove vocabulary is so a prediction must be made and then reapllied to glove to get a size 50 word embedding back
            
            loss += criterion(dec_output, target_tensor[di]) # similar words have lower loss
            dec_input = target_tensor[di]

    else:
        for di in range(target_length):
            dec_output, (dec_hidden, dec_context) = decoder.forward(dec_input, dec_hidden, dec_context)
            dec_output = list(embeddings.values())[torch.argmax(dec_output)]
            
            decoder_input = dec_output

            loss += criterion(dec_output, target_tensor[di])
            if torch.equal(decoder_input, EOS_TOKEN):
                break
        
    loss.requires_grad = True
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

# def training_iterations():
    
criterion = nn.MSELoss()
learning_rate = 0.001 
    
glove = Path("ASL_grammar_conversion/english_glove.txt")

embeddings = make_embeddings(embeddings_file=glove)
embeddings_length = len(embeddings)

encoder = EncoderRNN(50)
decoder = DecoderRNN(embeddings_length)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

print(train(sentenceToTensor(embeddings, "now where is the food dude?"), sentenceToTensor(embeddings, "dude where now the is food?"), 
      encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, embeddings))
