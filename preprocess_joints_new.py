import os

import mediapipe as mp
import numpy as np

from hand_cropping import crop_hand_joint

train_img = np.load("data/train_img.npy")
test_img = np.load("data/test_img.npy")
train_lab = np.load("data/train_lab.npy")
test_lab = np.load("data/test_lab.npy")

hands = mp.solutions.hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)
train_joints = []
n_train_lab = []
for img, lab in zip(train_img, train_lab):
    joint = crop_hand_joint(img, hands)
    if joint is None:
        continue
    train_joints.append(joint[0])
    n_train_lab.append(lab)

train_joints = np.concatenate(train_joints, axis=0)
n_train_lab = np.array(n_train_lab)

test_joints = []
n_test_lab = []
for img, lab in zip(test_img, test_lab):
    joint = crop_hand_joint(img, hands)
    if joint is None:
        continue
    test_joints.append(joint[0])
    n_test_lab.append(lab)

test_joints = np.concatenate(test_joints, axis=0)
n_test_lab = np.array(n_test_lab)

np.save("data/train_joints", train_joints)
np.save("data/train_lab_joint", n_train_lab)

np.save("data/test_joints", test_joints)
np.save("data/test_lab_joint", n_test_lab)
