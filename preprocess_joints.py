import os

import numpy as np

from hand_cropping import crop_hand_data

image_data = []
label_data = []
for c in ["A", "B", "C", "D", "E"]:
    for path in os.listdir(f"data/{c}"):
        if path == ".DS_Store":
            continue
        output = crop_hand_data(f"data/{c}/{path}")
        image_data.append(output)
        label_data.append([ord(path) - ord("a")] * output.shape[0])

image_data = np.concatenate(image_data)
label_data = np.concatenate(label_data)
indices = np.arange(len(label_data))
np.random.shuffle(indices)
image_data = image_data[indices]
label_data = label_data[indices]
np.save("data/n_joints", image_data)
np.save("data/n_labels", label_data)

# print(image_data)

# =======================================================================================================================================
# code from train_test_split.py

joints = np.load("data/n_joints.npy")
labels = np.load("data/n_labels.npy")

test_split = 0.2

train_len = int(len(joints) * (1 - test_split))
train_joints = joints[:train_len]
test_joints = joints[train_len:]
train_labels = labels[:train_len]
test_labels = labels[train_len:]

np.save("data/n_joints_train", train_joints.astype(np.float32))
np.save("data/n_joints_test", test_joints.astype(np.float32))
np.save("data/n_labels_train", train_labels.astype(np.int64))
np.save("data/n_labels_test", test_labels.astype(np.int64))
