from cv2 import data
import numpy as np
import cv2
import os
import random

chars = ['A','B','C','D','E']
database = []
for c in chars:
    for path in os.listdir(f'data/{c}'):
        if path == '.DS_Store':
            continue
        for img_path in os.listdir(f'data/{c}/{path}'):
            if img_path == '.DS_Store':
                continue
            image = cv2.imread(f'data/{c}/{path}/{img_path}')
            # scales/pads the images so that they're all shape (100, 100, 3):
            max_dim = np.argmax(image.shape[:2])
            scale_percent = 100 / image.shape[max_dim]
            width = round(image.shape[1] * scale_percent)
            height = round(image.shape[0] * scale_percent) 
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if max_dim == 0:
                image = cv2.copyMakeBorder(image, 0,0,0,100-image.shape[1],cv2.BORDER_CONSTANT,0)
            else:
                image = cv2.copyMakeBorder(image, 0,100-image.shape[0],0,0,cv2.BORDER_CONSTANT,0)
            database.append((image,ord(path)-ord('a')))

random.shuffle(database)

ind = [random.randint(0, 1000) for i in range(10)]
for i in ind: 
    print(database[i][0].shape)

imgs = []
labs = []
t_imgs = []
t_labs = []
train_split = 0.8
for i, (img, lab) in enumerate(database):
    if img.shape != (100,100,3):
        print('abnormal shape')
        print(img.shape)
    if i > train_split*len(database):
        t_imgs.append(img)
        t_labs.append(lab)
    else:
        imgs.append(img)
        labs.append(lab)
imgs = np.array(imgs)
labels = np.array(labs)
t_imgs = np.array(t_imgs)
t_labels = np.array(t_labs)
np.save('data/images', imgs)
np.save('data/labels', labels)
np.save('data/test_images', t_imgs)
np.save('data/test_labels', t_labels)