import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from data_processing import normalize_joints
from hand_cropping import crop_hand_joint
from sign_recogn_joint import RecogJoint
from signtotext import filter_text, sign_to_text


def videototext():
    model = RecogJoint()
    model.load_state_dict(
        torch.load("sign_recogn_joint_new", map_location=torch.device("cpu"))
    )
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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.45,
        )
        while True:

            def detect(texts, pred_scores, cap, model):
                suc, img = cap.read()
                if suc is None:
                    time.sleep(0.01)
                    return -1
                results = crop_hand_joint(img, hands)
                if results is None:
                    texts.append(" ")
                    pred_scores.append(0)
                    time.sleep(0.01)
                    return -1
                # print(crops.shape)
                results = normalize_joints(results[0])
                preds = model(torch.tensor(results).to("cpu")).detach().numpy()[0]
                # N, 26
                pred_scores.append(np.max(preds))
                text = alphabet[np.argmax(preds)]
                texts.append(text)
                print(text)
                cv2.imshow("image", img)
                # print(normalize(crops)[0])

            if detect(texts, pred_scores, cap, model) == -1:
                continue
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
        pred_scores = np.stack(pred_scores, axis=0)
        # print(pred_scores)
        # print(pred_scores.shape)
        text = sign_to_text(texts, pred_scores)
        filtered = filter_text(text)
        print(text)
        return filtered
