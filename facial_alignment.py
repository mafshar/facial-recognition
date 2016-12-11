#!/usr/bin/env python

import cv2
import os
import sys
import numpy as np

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv3/3.1.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv3/3.1.0_3/share/OpenCV/haarcascades/haarcascade_eye.xml')
    img = cv2.imread('./data/gt_db/s01/01.jpg')
    cv2.imshow('img', img)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imshow('img', roi_color) ## displays the face in the image
        cv2.waitKey(0)
