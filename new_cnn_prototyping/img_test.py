import cv2
import numpy as np

train_img = np.load("data/train_img.npy")
train_lab = np.load("data/train_lab.npy")
for i in range(10):
    cv2.imshow("image", train_img[i] / 255)
    print(train_lab[i])
    cv2.waitKey(0)
