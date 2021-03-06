#!/usr/bin/env python

import os
import sys
import urllib
import httplib
import cv2
import numpy as np

ACTOR_FILENAME = './data/faceScrub/facescrub_actors.txt'
ACTRESS_FILENAME = './data/faceScrub/facescrub_actresses.txt'

def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
    content.pop(0)
    return [s.strip().split() for s in content]

def download_transform_image(url, img_file_name, coords):
    coords = [int(v) for v in coords.strip().split(',')]
    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    download_command = 'wget -t 1 -O ' + img_file_name + ' ' + url
    delete_command = 'rm ' + img_file_name
    os.system(download_command)
    img = cv2.imread(img_file_name)
    if img is not None:
        crop_img = img[y1:y2, x1:x2]
        try:
            crop_img = cv2.resize(crop_img, (70, 70))
            cv2.imwrite(img_file_name, crop_img)
        except cv2.error:
            os.system(delete_command)
    else:
        os.system(delete_command)

def get_attributes(sample):
    if len(sample) == 7:
        curr_name = sample[0] + ' ' + sample[1]
        face_id = sample[3]
        url = sample[4]
        coords = sample[5]
    elif len(sample) == 8:
        curr_name = sample[0] + ' ' + sample[1] + ' ' + sample[2]
        face_id = sample[4]
        url = sample[5]
        coords = sample[6]
    return curr_name, face_id, url, coords

def generate_data(raw, starting_label=0):
    train_path = './data/training-actress'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    label = starting_label
    prev_name = raw[0][0] + ' ' + raw[0][1] ## first sample
    class_path = os.path.join(train_path, str(label))
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    for sample in raw:
        curr_name, face_id, url, coords = get_attributes(sample)
        if curr_name != prev_name:
            label += 1
            class_path = os.path.join(train_path, str(label))
            if not os.path.exists(class_path):
                os.makedirs(class_path)
        img_file_name = os.path.join(class_path, url.strip().split('/')[-1])
        download_transform_image(url, img_file_name, coords)
        prev_name = curr_name
            ## we're on a different class now

def rename_training_directories():
    train_path = os.path.abspath('./data/more_data')
    count = 1
    for folder in os.listdir(train_path):
        print folder
        if not os.path.isdir(os.path.join(train_path, folder)):
            continue # Not a directory
        if count < 10:
            new_name = "0"+ str(count)
            os.renames(os.path.join(train_path, folder), os.path.join(train_path, new_name))
        else:
            new_name = str(count)
            os.renames(os.path.join(train_path, folder), os.path.join(train_path, new_name))
        count+=1

def rename_training_files():
    train_path = os.path.abspath('./data/training-actress')
    for folder in os.listdir(train_path):
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(train_path,folder)):
            abs_folder = os.path.abspath(os.path.join(train_path,folder))
            count = 1
            for file_name in filenames:
                actual_name, extension = os.path.splitext(file_name)
                if count < 10:
                    new_name = "0" + str(count)
                    old_file_name = os.path.join(abs_folder,file_name)
                    new_part = new_name + extension
                    new_file_name = os.path.join(abs_folder, new_part)
                    new_file_name = os.path.abspath(new_file_name)
                    print new_file_name
                    os.renames(old_file_name, new_file_name)
                else:
                    new_name = str(count)
                    old_file_name = os.path.join(abs_folder,file_name)
                    new_part = new_name + extension
                    new_file_name = os.path.join(abs_folder, new_part)
                    new_file_name = os.path.abspath(new_file_name)
                    print new_file_name
                    os.renames(old_file_name, new_file_name)
                count += 1
