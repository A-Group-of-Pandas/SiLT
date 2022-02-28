import random
import re
from pathlib import Path


import numpy as np


def english_to_ASL(
    corpus_filename, non_ASL_filename, percent_for_training: float = 0.8
):
    """Turns a text corpus into pusedo-ASL grammar."""

    corpus_file = Path(corpus_filename)
    non_ASL_file = Path(non_ASL_filename)

    # reads the corpus
    with open(corpus_file, mode="r") as corpus:
        corpus = corpus.read().lower()

    with open(non_ASL_file, mode="r") as removable_words:
        removable_words = removable_words.read().lower()

    # removes non-end punctuation from the corpus
    mid_punc = """#$%&()*+,/:;<=>@[\]^_`"{|}~'"""
    corpus = corpus.translate(str.maketrans("", "", mid_punc))

    # splits and cleans the corpus
    truth_sentences = corpus.split(". ")
    truth_sentences = [sent.strip() for sent in truth_sentences]
    random.shuffle(truth_sentences)

    # randomly swaps around the words in each sentence
    input_sentences = [
        " ".join(random.sample(sent.split(" "), len(sent.split(" "))))
        for sent in truth_sentences
    ]

    # this splits our data into train inputs, test inputs and train truths, test truths then turns it all into a string with each new scrambled sentence on a new line with a period at the end
    data_split = round(len(truth_sentences) * percent_for_training)
    X_input_sentences = "\n".join(input_sentences[:data_split])
    Y_input_sentences = "\n".join(input_sentences[data_split:])
    X_truth_sentences = "\n".join(truth_sentences[:data_split])
    Y_truth_sentences = "\n".join(truth_sentences[data_split:])

    # removes non ASL words from the input sentences
    removable_words = removable_words.split("\n")
    for word in removable_words:
        X_input_sentences = re.sub(f" {word} ", " ", X_input_sentences)
        Y_input_sentences = re.sub(f" {word} ", " ", Y_input_sentences)

    return X_input_sentences, Y_input_sentences, X_truth_sentences, Y_truth_sentences


non_ASL_path = Path("ASL_grammar_generation/non_ASL_words.txt")
corpus_path = Path("ASL_grammar_generation/lung_cancer.txt")

(X_input_sentences, Y_input_sentences, X_truth_sentences, Y_truth_sentences) = english_to_ASL(corpus_filename=corpus_path, non_ASL_filename=non_ASL_path)

with open("./x_input_data.txt", mode="w") as _file:
    _file.write(X_input_sentences)
with open("./y_input_data.txt", mode="w") as _file:
    _file.write(Y_input_sentences)

with open("./x_truth_data.txt", mode="w") as _file:
    _file.write(X_truth_sentences)
with open("./y_truth_data.txt", mode="w") as _file:
    _file.write(Y_truth_sentences)
