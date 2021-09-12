import os

import cv2
import mediapipe as mp
import numpy as np

from data_processing import resize_crop, resize_pad


def crop_hand_data(image_folder, hand_num=1):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=True, max_num_hands=hand_num, min_detection_confidence=0.5
    )
    files = [file for file in os.listdir(image_folder) if not "DS_Store" in file]
    RGBimgs = np.array(
        [
            resize_crop(
                cv2.flip(
                    cv2.cvtColor(
                        cv2.imread(f"{image_folder}/{file}"), cv2.COLOR_BGR2RGB
                    ),
                    1,
                )
            )
            for file in files
        ]
    )
    labs = [ord(file.split(".")[0]) - ord("a") for file in files]
    all_results = []
    for img in RGBimgs:
        results = hands.process(img)
        if not results.multi_hand_landmarks:
            continue
        # (#hands, #landmarks, 3)
        results_list = np.array(
            [
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
                for hand_lms in results.multi_hand_landmarks
            ]
        )
        all_results.append(results_list)

    return np.array(all_results), labs


# =======================================================================================================================================
# FUNCTIONS FROM HAND_CROPPER
def crop_hand_joint(img, hands):
    # margin gives some space between the tips of fingers and the bounding box (bbox) measured in pixels
    # plays recording from camera and processes each image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if not results.multi_hand_landmarks:
        return None
    results = np.array(
        [
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
            for hand_lms in results.multi_hand_landmarks
        ]
    )
    del imgRGB, img
    return results[None, ...].astype(np.float32)


def crop_hand_cnn(img, hands, margin=0.07):
    # plays recording from camera and processes each image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    cropped_results = []
    if not results.multi_hand_landmarks:
        return None
    for handLms in results.multi_hand_landmarks:
        landmark_listx = []
        landmark_listy = []
        for lm in handLms.landmark:
            h, w, c = img.shape
            # here we multiply by the width and height because the landmarks are auto-normalized based on
            # the width and height of the displayed image
            landmark_listx.append((lm.x * w))
            landmark_listy.append((lm.y * h))
        end = (
            int(max(landmark_listx)) + int(margin * w),
            int(max(landmark_listy)) + int(margin * h),
        )
        start = (
            max(0, int(min(landmark_listx)) - int(margin * w)),
            max(0, int(min(landmark_listy)) - int(margin * h)),
        )
        cropped_img = img[start[1] : end[1], start[0] : end[0]]
        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            continue
        # print(cropped_img.shape)
        cropped_results.append(resize_pad(cropped_img))
    if len(cropped_results) == 0:
        return None
    return np.array(cropped_results, dtype=np.float32)


# =======================================================================================================================================


#     while True:
#         # plays recording from camera and processes each image
#         success, img = cap.read()
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(imgRGB)
#         # print(results.multi_hand_landmarks)

#         if results.multi_hand_landmarks:
#             for handLms in results.multi_hand_landmarks:
#                 landmark_listx = []
#                 landmark_listy = []
#                 for lm in handLms.landmark:
#                     h, w, c = img.shape

#                     # here we multiply by the width and height because the landmarks are auto-normalized based on
#                     # the width and height of the displayed image
#                     landmark_listx.append((lm.xw))
#                     landmark_listy.append((lm.yh))
# [12:22 PM] kw-0: # Uncomment this for landmarking the joints:

#                 cx, cy = int(lm.xw), int(lm.yh)   # the landmarks are auto-normalized by the width and height, so we have to multiply them back to scale to put them on img

#                 if id == False:
#                     cv2.circle(img, (cx,cy), 3, (255, 166, 48), cv2.FILLED)
#                 print(handlms.landmark)

#                 mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


#                 # creating the starts and ends of the box around the hand (+ some margin)
#                 start = (int(max(landmark_listx))+margin, int(max(landmark_listy))+margin)
#                 end = (int(min(landmark_listx))-margin, int(min(landmark_listy))-margin)

#                 img = cv2.rectangle(img, start, end, color=(255, 166, 48), thickness=thickness_of_bbox)

#         cv2.imshow("image", img)


#         if cv2.waitKey(1) & 0xFF == ord(' '):
#             break

#     cv2.destroyAllWindows()

#     # saves the image as a jpg and crops it down to just the hand (the +thickness+1 is to remove the box and the gradient area between the box and actual hand)
#     # the [:] at the beginning is to apply this crop to all color channels
#     img = img[:][end[1]+(thickness_of_bbox+1):start[1]-(thickness_of_bbox-1),
#                                     end[0]+(thickness_of_bbox+1):start[0]-(thickness_of_bbox-1)]
#     return
