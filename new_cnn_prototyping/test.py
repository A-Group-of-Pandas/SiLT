import os

import cv2
import mediapipe as mp
import numpy as np
import torch
from img_proc_help import (crop_hand_cnn, normalize, normalize_joints,
                           resize_pad)
from sign_recogn_cnn import SignRecogCNN
from torch import nn


def read_test():
    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )
    imgs = []
    labs = []
    joints = []
    for k in os.listdir("test"):
        if "DS" in k:
            continue
        img = cv2.imread(f"test/{k}")
        img, joint = crop_hand_cnn(img, hands, random=False)
        joints.append(joint[0, 0])
        imgs.append(img[0])
        labs.append(ord(k.split(".")[0]) - ord("a"))
    imgs = np.array(imgs)
    labs = np.array(labs)
    joints = np.array(joints)
    return (
        torch.tensor(normalize(imgs)).to("cpu"),
        torch.tensor(normalize_joints(joints)).to("cpu"),
        torch.tensor(labs).to("cpu"),
    )


imgs, joints, labs = read_test()


model = SignRecogCNN()
model.load_state_dict(
    torch.load("sign_recogn_joint_cnn-4", map_location=torch.device("cpu"))
)
loss_fn = nn.CrossEntropyLoss()

model.eval()
with torch.no_grad():
    preds = model(imgs, joints)
    loss = loss_fn(preds, labs)
    acc = (
        torch.mean((torch.argmax(preds.detach(), dim=1) == labs).float())
        .detach()
        .item()
    )
    print(loss, acc)
