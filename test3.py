import gc

import cv2
import mediapipe as mp

from hand_cropping import crop_hand_cnn

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.2,
)
while True:
    suc, img = cap.read()
    if not suc:
        continue
    results = crop_hand_cnn(img, hands)
    if results is None:
        continue
    cv2.imshow("image", results[0] / 225)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
    del img, results
    gc.collect()
