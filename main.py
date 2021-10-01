import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    success, img = video.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    print(results.multi_hand_landmarks)


    cv2.imshow("Hand Detector", img)
    
    cv2.waitKey(1)