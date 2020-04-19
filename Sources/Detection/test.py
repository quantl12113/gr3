import numpy as np
import cv2
import os
path="./face1"

face_cascade = cv2.CascadeClassifier('../../opencv/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture("test1.mp4")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
i=0
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        test=img[y:y+h, x:x+w]
        resized = cv2.resize(test, (128, 128), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(path , 'test'+str(i)+'.png'), resized)
        print ("",i)
        i=i+1
        print(i)