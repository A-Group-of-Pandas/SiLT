import random
from collections import Counter

import cv2
import numpy as np
import torch
from numpy.core.defchararray import count

from data_processing import normalize
from sign_recog_cnn import SignRecogCNN

# images = np.load("data/images.npy")
# labels = np.load("data/labels.npy")
# print(images.shape)

# indx = [random.randint(0, 5000) for i in range(10)]
# images = images[indx]
# for i in range(10):
#     cv2.imshow('image',images[i])
#     cv2.waitKey(0)
# images = normalize(images)
# print(images.shape)
# labels = labels[indx]
# alphabet = "abcdefghijklmnopqrstuvwxyz"
# for i in range(10):
#     print(alphabet[labels[i]], end="")
#     #cv2.imshow('image', i)
#     #cv2.waitKey(0)
# print()

# model = SignRecogCNN()
# model.load_state_dict(torch.load('sign_recogn_cnn_larger', map_location=torch.device('cpu')))
# model.eval()
# with torch.no_grad():
#     images = torch.tensor(images).to('cpu')
#     encodings = []
#     preds = model(images)

# converts each sign to appropriate text.
def sign_to_text(text, confidence_scores, interv=10, threshold=0.7):
    """Converts a list of signs ()
    and returns their corresponding text."""
    # (T, 26)
    ls_str = []
    interval_str = []
    for i, letter in enumerate(text):
        # print((i, letter))
        interval_str.append((i, letter))
        if len(interval_str) == interv:
            # print(interval_str)
            letters_descending = Counter(
                [letter for index, letter in interval_str]
            ).most_common()
            # print(letters_descending)
            for l, _ in letters_descending:
                # print([index for index, element in interval_str if element == l])
                # print(confidence_scores[[index for index, element in interval_str if element == l]])
                if (
                    np.mean(
                        confidence_scores[
                            [index for index, element in interval_str if element == l]
                        ]
                    )
                    > threshold
                ):
                    ls_str.append(l)
                    break
            interval_str = []
    return ls_str


def filter_text(ls_str, threshold=3):
    letters = []
    char = ls_str[0]
    count = 1
    for character in ls_str:
        if character != char:
            if count > threshold:
                if not char == " ":
                    letters.append(char)
                char = character
                count = 0
        count += 1
    if count > threshold:
        letters.append(char)
    return "".join(letters)


# text = list('hhhhhhhhheeeeeeellllllllllll llllllllllooooooooo         mmmmmmmmmmmmyyyyyyyyyyy nnnnnnnnnnaaaaaaaaaaaammmmmmmmmeeeeeeeeeee             iiiiiiiisssssssssss         ppppppppppiiiiiiiiiikkkkkkkkkkkaaaaaaaaaa')
# confidence = np.ones((len(text)))
# print(sign_to_text(text,confidence))
# print(sign_to_text(preds))
