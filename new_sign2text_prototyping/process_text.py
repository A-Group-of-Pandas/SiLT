import re

import numpy as np
from seq2seq import Seq2Seq


def read_wiki():
    path_to_wikipedia = "wikipedia2text-extracted.txt"  # update this path if necessary
    with open(path_to_wikipedia, "rb") as f:
        wikipedia = f.read().decode().lower()
    wikipedia = re.sub(r"[^a-z ]+", "", wikipedia)
    return wikipedia


def vectorize(text):
    vec = []
    text = text.lower()
    for let in text:
        if let == " ":
            vec.append(26)
        else:
            vec.append(ord(let) - ord("a"))
    vec.append(28)
    vecs = []
    for num in vec:
        zero = np.zeros((29))
        zero[num] = 1
        vecs.append(zero)
    return np.array(vecs)


def gen_data(length, rep):
    source = read_wiki()
    idx = np.arange(len(source))
    np.random.shuffle(idx)
    for i in idx:
        text = source[i : i + length]
        t_text = []
        for let in text:
            for i in range(rep):
                if np.random.uniform(0, 1) > 0.8:
                    randnum = np.random.randint(0, 27)
                    if randnum == 26:
                        t_text.append(" ")
                    else:
                        t_text.append(chr(randnum + ord("a")))
                else:
                    t_text.append(let)
        vecs = []

        def let2num(let):
            if let == " ":
                return 26
            else:
                return ord(let) - ord("a")

        for let in t_text:
            letvec = np.zeros(29)
            letvec[let2num(let)] = 1
            vecs.append(letvec)
        labels = [let2num(let) for let in text]
        labels.append(28)
        yield np.array(vecs), np.array(labels)


def gen_batch(batch_size=128):
    batch_vecs = []
    batch_lets = []
    end_let = np.zeros((29))
    end_let[28] = 1.0
    while True:
        length = np.random.randint(3, 50)
        rep = np.random.randint(5, 40)
        data_gen = gen_data(length, rep)
        for i, (vecs, lets) in enumerate(data_gen):
            # print(vecs[0].shape)
            batch_vecs.append(vecs[:, None, :])
            batch_lets.append(lets[:, None])
            if i + 1 == batch_size:
                break
        batch_vecs = np.concatenate(batch_vecs, axis=1)
        batch_lets = np.concatenate(batch_lets, axis=1)
        yield batch_vecs, batch_lets
        batch_vecs = []
        batch_lets = []
        max_let_len = 0
        max_len = 0
