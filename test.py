"""
Author: Yuhan (Will) Chen
"""

from keras.models import Model, load_model
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
NUM_COORDS = 2
TASK_NAME = 'fine_phone'
TEST_SPLIT = 0.1

INDEX_FILE = 'index_file.txt'
MODEL_PATH = '../vgg16_fine_phone_weights.h5'


def load_index(index_file):
    f = open(index_file, 'rb')
    indices = f.readlines()
    index_list = [int(idx.strip()) for idx in indices]
    return index_list


def load_labels(file_path):
    f = open(file_path, 'rb')
    content = f.readlines()
    label_dict = {}
    for label in content:
        label_parts = label.strip().split(' ')
        label_dict[label_parts[0]] = (label_parts[1], label_parts[2])
    return label_dict


def partition_data(X, Y):
    """

    :param X:
    :param Y:
    :return: train images, train labels, test images, test labels
    """
    n_samples = X.shape[0]
    idx_list = load_index(INDEX_FILE)

    test_idx = idx_list[:int(n_samples*TEST_SPLIT)]
    train_idx = idx_list[int(n_samples*TEST_SPLIT):]
    return X[train_idx,:,:,:], Y[train_idx,:], X[test_idx,:,:,:], Y[test_idx,:]


def load_data(src_path):

    file_list = os.listdir(src_path)
    image_name_list = [fname for fname in file_list if fname.endswith('.jpg')]
    num_images = len(image_name_list)
    print '-- This set has {} images.'.format(num_images)
    X = np.zeros((num_images, IMG_H, IMG_W, NUM_CHANNELS))
    Y = np.zeros((num_images, 2))
    # read images and labels
    label_dict = load_labels(LABEL_FILE)
    for i, image_name in enumerate(image_name_list):
        image_path = os.path.join(src_path, image_name)
        image = cv2.imread(image_path, 1)
        #image = process_image(image)
        image = cv2.resize(image, (IMG_H, IMG_W)) - MEAN_PIXEL
        X[i, :, :, :] = image

        c1, c2 = label_dict[image_name]
        Y[i,0], Y[i,1] = c1, c2
    return partition_data(X,Y)


def main():
    # load data
    X_train, Y_train, X_test, Y_test = load_data(TRAIN_DIR)

    #load model
    model = load_model(MODEL_PATH)

    # check results
    predictions = model.predict(x=X_test)
    print predictions


if __name__ == '__main__':
    main()