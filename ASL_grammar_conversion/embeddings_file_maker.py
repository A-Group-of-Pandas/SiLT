import numpy as np
from pathlib import Path
import textwrap as tw

GloVe_file = Path("glove/glove.6B.50d.txt")
eng_words = Path("ASL_grammar_conversion/words.txt")
non_ASL_file = Path("ASL_grammar_conversion/non_ASL_words.txt")


with open(non_ASL_file, mode="r") as non_ASL:
    non_ASL = non_ASL.read().split()
    
with open(eng_words, mode="r") as eng_words:
    eng_words = eng_words.read().split()

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

embeddings = make_embeddings(GloVe_file)
# a = list(embeddings.keys())
del_list = []
i = 0
for word in embeddings:
    if i % 5000 == 0:
        print(f"word: {word} \n num: {i}")
    if word not in eng_words:
        del_list.append(word)
    i+=1

# print(del_list)
for word in del_list:
    del embeddings[word]
    
wrapper = tw.TextWrapper(width=10000)
english_file = "ASL_grammar_conversion/english_glove.txt"
with open(english_file, mode="w") as english:
    for key in embeddings:
        value = str(embeddings[key])
        english.write(key)
        english.write(" " + str(wrapper.wrap(value)).replace('[', '').replace(']', '').replace("'", "").strip())
        english.write('\n')

