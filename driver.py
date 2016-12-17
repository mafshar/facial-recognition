#!/usr/bin/env python

import os
import facial_alignment
import data_spliter
import training_retriever


if __name__ == '__main__':
<<<<<<< HEAD
    actor_data = training_retriever.read_file(training_retriever.ACTOR_FILENAME)
    actress_data = training_retriever.read_file(training_retriever.ACTRESS_FILENAME)
    i = 0
    for j in range(len(actress_data)):
        if actress_data[j][0] + ' ' + actress_data[j][1] != 'Holly Marie' :
            i = j
        else:
            break
    actress_data = actress_data[i:]
    actress_data.pop(0)
    # print actress_data[0]
    training_retriever.generate_data(actress_data, 72)
    # actor_data = training_retriever.read_file(training_retriever.ACTOR_FILENAME)
    # training_retriever.generate_data(actor_data)
    training_retriever.rename_training_directories()
    training_retriever.rename_training_files()


    # if facial_alignment.RESIZE_FLAG:
    #     train_path = './data/training'
    #     for path, dirs, files in os.walk(train_path):
    #         for f in files:
    #             img_file_name = None
    #             if f.endswith('.jpg'):
    #                 img_file_name = os.path.join(path, f)
    #                 img = cv2.imread(img_file_name)
    #                 new_img = cv2.resize(img, (100, 100))
    #                 cv2.imwrite(img_file_name, new_img)
    #
    # else:
    #     old_path = './data/gt_db'
    #     new_path_train = './data/training'
    #     if not os.path.exists(new_path_train):
    #         os.makedirs(new_path_train)
    #     for path, dirs, files in os.walk(old_path):
    #         # print path
    #         class_num = path.strip().split('/')[-1][1:]
    #         class_dir = os.path.join(new_path_train, class_num)
    #         if not os.path.exists(class_dir):
    #             os.makedirs(class_dir)
    #         for f in files:
    #             input_img_file_name = None
    #             if f.endswith('.jpg'):
    #                 input_img_file_name = os.path.join(path, f)
    #                 output_img_file_name = os.path.join(class_dir, f)
    #                 facial_alignment.detect_align_face(input_img_file_name, output_img_file_name)
