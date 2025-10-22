import cv2
import mediapipe as mp 
import time 
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands = 4)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handlms.landmark):
                #print(id, lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                #if id == 4:
                #    cv2.circle(img, (cx,cy), 15, (0,255,0), cv2.FILLED)



    cv2.imshow("Image", img)
    cv2.waitKey(1)
