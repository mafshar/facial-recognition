#!/usr/bin/env python

import os
import sys
import urllib
import cv2
import numpy as np

ACTOR_FILENAME = './data/faceScrub/facescrub_actors.txt'
ACTRESS_FILENAME = './data/faceScrub/facescrub_actoresses.txt'

def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
    content.pop(0)
    return [s.strip().split() for s in content]

def download_transform_image(url, img_file_name):
    print img_file_name
    urllib.urlretrieve(url, img_file_name)

    return

def generate_data(raw, starting_label=0):
    train_path = './data/training'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    label = starting_label
    prev_name = raw[0][0] + ' ' + raw[0][1] ## first sample
    class_path = os.path.join(train_path, str(label))
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    img_num = 0
    for sample in raw:
        curr_name = sample[0] + ' ' + sample[1]
        url = sample[4]
        coords = sample[5]
        if curr_name == prev_name:
            img_num += 1
            img_file_name = os.path.join(class_path, str(img_num) + '.jpg')
            download_transform_image(url, img_file_name)
            ## we're on the same class
        else:
            pass
            ## we're on a different class now
