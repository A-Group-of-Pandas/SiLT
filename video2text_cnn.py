import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from data_processing import normalize
from hand_cropping import crop_hand_cnn
from sign_recog_cnn import SignRecogCNN
from signtotext import filter_text, sign_to_text

model = SignRecogCNN()
model.load_state_dict(torch.load("sign_recogn_cnn", map_location=torch.device("cpu")))
model.eval()
st = time.time()
alphabet = "abcdefghijklmnopqrstuvwxyz"
with torch.no_grad():
    cap = cv2.VideoCapture(0)
    texts = []
    pred_scores = []
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.35,
    )
    while True:

        def detect(texts, pred_scores, cap, model):
            suc, img = cap.read()
            if suc is None:
                return -1
            crops = crop_hand_cnn(img, hands, margin=0.05)
            if crops is None:
                texts.append(" ")
                pred_scores.append(7)
                time.sleep(0.01)
                return -1
            crop = normalize(crops[0:1])
            preds = model(torch.tensor(crop)).detach().numpy()[0]
            # N, 26
            pred_scores.append(np.max(preds))
            text = alphabet[np.argmax(preds)]
            texts.append(text)
            print(text)
            cv2.imshow("image", crops[0] / 255)
            # print(normalize(crops)[0])

        if detect(texts, pred_scores, cap, model) == -1:
            continue
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    pred_scores = np.stack(pred_scores, axis=0)
    text = sign_to_text(texts, pred_scores)
    filtered = filter_text(text)
    print(text)
    print(filtered)

    # print(sign_to_text(texts, pred_scores))
