import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.45)
mpDraw = mp.solutions.drawing_utils

margin = 40


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmark_listx = []
            landmark_listy = []
            for lm in handLms.landmark:
                h, w, c = img.shape

                landmark_listx.append((lm.x*w))
                landmark_listy.append((lm.y*h))
                
                """
                # Uncomment this for landmarking the joints:

                cx, cy = int(lm.x*w), int(lm.y*h)   # the landmarks are auto-normalized by the width and height, so we have to multiply them back to scale to put them on img

                if id == False:
                cv2.circle(img, (cx,cy), 3, (255, 166, 48), cv2.FILLED)
                print(handlms.landmark)
    
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                """

            start = (int(max(landmark_listx))+margin, int(max(landmark_listy))+margin)
            end = (int(min(landmark_listx))-margin, int(min(landmark_listy))-margin)

            img = cv2.rectangle(img, start, end, color=(255, 166, 48), thickness=2)

    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cv2.imwrite('image.jpg', img[:][end[1]:start[1], end[0]:start[0]])

cv2.destroyAllWindows()