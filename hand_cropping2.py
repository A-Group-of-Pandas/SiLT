import cv2
import mediapipe as mp
import numpy as np

f = cv2.cvtColor(cv2.imread("handtest.png"), cv2.COLOR_BGR2RGB)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.45,
)
mpDraw = mp.solutions.drawing_utils

results = hands.process(f)


results_list = [
    [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
    for hand_lms in results.multi_hand_landmarks
]

print(np.array(results_list).shape)

# print(results.multi_hand_landmarks)
