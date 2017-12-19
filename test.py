"""
Author: Yuhan (Will) Chen
"""

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
# from keras.utils.np_utils import to_categorical
import numpy as np
# import glob
import os
import cv2
import random

IMG_H, IMG_W, NUM_CHANNELS = 224, 224, 3
MEAN_PIXEL = np.array([104., 117., 123.]).reshape((1,1,3))
TRAIN_DIR = '../find_phone'
LABEL_FILE = '../find_phone/labels.txt'
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_COORDS = 2
TASK_NAME = 'fine_phone'

INDEX_FILE = 'index_file.txt'


def load_index(index_file):
    f = open(index_file, 'rb')
    indices = f.readlines()
    index_list = [int(idx.strip()) for idx in indices]
    print index_list


def main():
    load_index(INDEX_FILE)


if __name__ == '__main__':
    main()