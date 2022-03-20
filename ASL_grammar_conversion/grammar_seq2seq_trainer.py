from grammar_seq2seq import *
import string
import torch
import random
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time

embeddings_dims = 50

SOS_TOKEN = torch.zeros(embeddings_dims)
EOS_TOKEN = torch.ones(embeddings_dims)

teacher_forcing_ratio = 0.65

def make_pairs(inputs, truths):
    return np.random.shuffle(list(zip(inputs, truths)))

def sentenceToTensor(embeddings, sentence):
    # embeddings is a dict
    sent = sentence.lower().strip()
    sent = sent.translate(str.maketrans("", "", string.punctuation))
    tensor = [embeddings[word] if word in embeddings else torch.ones(embeddings_dims)*0.5 for word in sent.split()]
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
        enc_output, (enc_hidden, enc_context) = encoder.forward(input_tensor[ei].view(1, 1, -1), enc_hidden, enc_context)
    
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

def train_iters(epochs, embeddings, inputs, truths, encoder, decoder, log_every=10, learning_rate=0.001):
    start_time = time.time()
    # log_every = 1

    # inputs = [sentenceToTensor(embeddings, sent) for sent in inputs]
    # truths = [sentenceToTensor(embeddings, sent) for sent in truths]
    # pairs = make_pairs(inputs, truths)
    
    # loss related things
    criterion = nn.MSELoss()
    logged_loss = 0
    avg_losses = torch.zeros((int(np.floor(epochs/log_every))))

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    print("started training")

    for epoch in range(1, epochs+1):
        input = sentenceToTensor(embeddings, inputs[epoch])
        truth = sentenceToTensor(embeddings, truths[epoch])
        
        # helps track time for every set of epochs we log and print
        if epoch % log_every == 0 or epoch==0:
            start_time = time.time()
                      
        loss = train(input, truth, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, embeddings)
        logged_loss += loss
        # print(logged_loss)
        
        # adds average loss of that section of epochs to a list for plotting and prints some useful info
        if epoch % log_every==0:
            avg_losses[int(epoch/log_every)-1] = logged_loss/log_every
            
            print(f"Epoch: {epoch} ({epoch/epochs}%) \n\t time elapsed: {(time.time() - start_time)} \n\t avg loss: {logged_loss/log_every} \n")
            logged_loss = 0


    return encoder.state_dict(), decoder.state_dict(), avg_losses

EPOCHS = 113
glove = Path("/Users/kylenwilliams/Desktop/projects/SiLT/ASL_grammar_conversion/english_glove.txt")

# make embeddings
embeddings = make_embeddings(embeddings_file=glove)

encoder = EncoderRNN(embeddings_dims)
decoder = DecoderRNN(embeddings_dims)

with open("/Users/kylenwilliams/Desktop/projects/SiLT/ASL_grammar_conversion/x_input_data.txt", mode="r") as inputs:
    inputs = inputs.readlines()
    
with open("/Users/kylenwilliams/Desktop/projects/SiLT/ASL_grammar_conversion/x_truth_data.txt", mode="r") as truths:
    truths = truths.readlines()

# don't forget to do validation

encoder_params, decoder_params, losses = train_iters(EPOCHS, embeddings, inputs, truths, encoder, decoder)

torch.save(encoder_params, "/Users/kylenwilliams/Desktop/projects/SiLT/ASL_grammar_conversion/encoder_params.pt")
torch.save(decoder_params, "/Users/kylenwilliams/Desktop/projects/SiLT/ASL_grammar_conversion/decoder_params.pt")

fig, ax = plt.subplots()
plt.plot(list(range((int(np.floor(EPOCHS/10))))), losses)
plt.title("Loss")
plt.show()
