import os
import random

import cv2
import numpy as np
from cv2 import data

from data_processing import resize

chars = ["A", "B", "C", "D"]
train_database = []
test_database = []
for c in chars:
    for path in os.listdir(f"data/{c}"):
        if path == ".DS_Store":
            continue
        for img_path in os.listdir(f"data/{c}/{path}"):
            if img_path == ".DS_Store":
                continue
            image = cv2.imread(f"data/{c}/{path}/{img_path}")
            # scales/pads the images so that they're all shape (100, 100, 3):
            image = resize(image)
            train_database.append([image, ord(path) - ord("a")])

for path in os.listdir(f"data/E"):
    if path == ".DS_Store":
        continue
    for img_path in os.listdir(f"data/{c}/{path}"):
        if img_path == ".DS_Store":
            continue
        image = cv2.imread(f"data/{c}/{path}/{img_path}")
        # scales/pads the images so that they're all shape (100, 100, 3):
        image = resize(image)
        test_database.append([image, ord(path) - ord("a")])

random.shuffle(train_database)
random.shuffle(test_database)

imgs = []
labs = []
for img, label in train_database:
    imgs.append(img)
    labs.append(label)
t_imgs = []
t_labs = []
for img, label in test_database:
    t_imgs.append(img)
    t_labs.append(label)

imgs = np.array(imgs)
labels = np.array(labs)
t_imgs = np.array(t_imgs)
t_labels = np.array(t_labs)
np.save("data/images", imgs)
np.save("data/labels", labels)
np.save("data/test_images", t_imgs)
np.save("data/test_labels", t_labels)
