import random
from pathlib import Path

corpus_path = Path("ASL_grammar_generation/wikipedia2text-extracted.txt")

# x and y here refer to training and testing respectively
corpus_X_input_path = Path("ASL_grammar_generation/x_input_data.txt")
corpus_Y_input_path = Path("ASL_grammar_generation/y_input_data.txt")

corpus_X_truth_path = Path("ASL_grammar_generation/x_truth_data.txt")
corpus_Y_truth_path = Path("ASL_grammar_generation/y_truth_data.txt")

split_value = 0.8

# reads the corpus
with open(corpus_path, mode="r") as corpus:
    corpus = corpus.read().lower()

# removes new lines and apostrophes from the corpus
removable_punc = """"#$%&'()*+,-/:;<=>@[\]^_`{|}~'"""
corpus = corpus.translate(str.maketrans("", "", removable_punc))


truth_sentences = corpus.split(". ")
random.shuffle(truth_sentences)

# randomizes the words in each sentence and saves it as a string where each new word-randomized sentence is on a newline
input_sentences = [
    " ".join(random.sample(sent.split(" "), len(sent.split(" "))))
    for sent in truth_sentences
]

# this splits our data into train inputs, test inputs and train truths, test truths then turns it all into a string with each new scrambled sentence on a new line with a period at the end
data_split = round(len(truth_sentences) * 0.8)

X_input_sentences = ".\n".join(input_sentences[:data_split])
Y_input_sentences = ".\n".join(input_sentences[data_split:])

X_truth_sentences = ".\n".join(truth_sentences[:data_split])
Y_truth_sentences = ".\n".join(truth_sentences[data_split:])

with open(corpus_X_input_path, mode="x") as _file:
    _file.write(X_input_sentences)
with open(corpus_Y_input_path, mode="x") as _file:
    _file.write(Y_input_sentences)

with open(corpus_X_truth_path, mode="x") as _file:
    _file.write(X_truth_sentences)
with open(corpus_Y_truth_path, mode="x") as _file:
    _file.write(Y_truth_sentences[data_split:])
