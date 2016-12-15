#!/usr/bin/env python
'''
Module to generate a random subsample of images
for training -- should only be used to *replace*
existing './data/randPairsDevTrain.txt' file
'''

import os
import numpy as np

FILENAME = './data/pairsDevTrain.txt'

def split_train(filename):
    with open(FILENAME) as f:
        content = f.readlines()
    content = [s for s in content]
    count = 1101
    same_sample = content[1:count]
    notsame_sample = content[count:]
    same_ndx = np.random.randint(0, len(same_sample), size=500)
    notsame_ndx = np.random.randint(0, len(notsame_sample), size=500)
    new_data_file = ''
    for ndx in same_ndx:
        new_data_file += same_sample[ndx]
    for ndx in notsame_ndx:
        new_data_file += notsame_sample[ndx]
    fh = open('./data/randPairsDevTrain.txt', 'w')
    fh.write(new_data_file)
