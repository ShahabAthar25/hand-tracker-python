import cv2

video = cv2.VideoCapture(0)

while True:
    success, img = video.read()

    cv2.imshow("Hand Detector", img)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break