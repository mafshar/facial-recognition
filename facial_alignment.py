#!/usr/bin/env python

import cv2
import os
import sys
import glob
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv3/3.1.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv3/3.1.0_4/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
RESIZE_FLAG = True

'''
def align_image(im_color, im_gray):
    #get height and width
    sz = im_gray.shape
    height = sz[0]
    width = sz[1]

    #set up aligned image matrix
    im_aligned = np.zeros((height, width,3), dtype=np.uint8)

    #copy over red
    im_aligned[:,:,2] = im_color[:,:,2]

    warp_mode = cv2.MOTION_HOMOGRAPHY

    warp_matrix = np.eye(3,3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)

    for i in xrange(0,2) :
        (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im_color[:,:,2]), get_gradient(im_color[:,:,i]),warp_matrix, warp_mode, criteria)

        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use Perspective warp when the transformation is a Homography
            im_aligned[:,:,i] = cv2.warpPerspective(im_color[:,:,i], warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use Affine warp when the transformation is not a Homography
            im_aligned[:,:,i] = cv2.warpAffine(im_color[:,:,i], warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        # print warp_matrix

    # Show final output
    # cv2.imshow("Color Image", im_color)
    # cv2.imshow("Aligned Image", im_aligned)
    # cv2.waitKey(0)
    return im_aligned
#helper function for findTransformECC (line 31)
def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad
def alignment():
    # iterate through s01 - s50
    for setRange in range(2,51):
        set_index = 's' + str(setRange)
        for image in range(1, 16):
            image_index = None
            if image < 10:
                image_index = '0' + str(image)
            else:
                image_index = str(image)

            img_str = './data/gt_db/' + set_index + '/' + image_index + '.jpg'

            img = cv2.imread(img_str)
            gray = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                #okay kew we've localized the face, now we have to align it
                print 'Aligning image: ' +  set_index + '/'+ image_index
                aligned = align_image(roi_color, roi_gray)
                print 'Finished alignment'
                img_save = './data/gt_db/' + set_index + '/' + image_index + '_aligned.jpg'
                cv2.imwrite(img_save, aligned)
'''

def detect_align_face(input_img_file, output_img_file):
    img = cv2.imread(input_img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # if faces:
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        face = cv2.resize(roi_color, (70, 70))
        cv2.imwrite(output_img_file, face)
    # else:
    #     print input_img_file
