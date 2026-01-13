import numpy as np
import cv2

cap = cv2.VideoCapture("./data/video.mp4")

haar = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
# https://github.com/opencv/opencv/tree/master/data/haarcascades

def face_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)

    return img

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = face_detect(frame)

    cv2.imshow('Object_detect', frame)

    if cv2.waitKey(40) == 27: # 27 es esc key
        break

cv2.destroyAllWindows()
cap.release()