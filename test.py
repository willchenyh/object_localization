"""
Author: Yuhan (Will) Chen
"""

from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
# from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import preprocess_input
import numpy as np
# import glob
import os
import cv2
import random

IMG_H, IMG_W, NUM_CHANNELS = 224, 224, 3
# MEAN_PIXEL = np.array([104., 117., 123.]).reshape((1,1,3))
TRAIN_DIR = '../find_phone'
LABEL_FILE = '../find_phone/labels.txt'
NUM_COORDS = 2
RADIUS = 0.05


TEST_SPLIT = 0.1
INDEX_FILE = 'index_file.txt'
MODEL_PATH = '../find_phone_vgg16_weights.h5'


def load_index(index_file):
    f = open(index_file, 'rb')
    indices = f.readlines()
    index_list = [idx.strip() for idx in indices]
    return index_list


def load_labels(file_path):
    f = open(file_path, 'rb')
    content = f.readlines()
    label_dict = {}
    for label in content:
        label_parts = label.strip().split(' ')
        label_dict[label_parts[0]] = (float(label_parts[1]), float(label_parts[2]))
    return label_dict


def get_partitions():
    image_list = load_index(INDEX_FILE)
    n_samples = len(image_list)
    test_set = image_list[:int(n_samples * TEST_SPLIT)]
    train_set = image_list[int(n_samples * TEST_SPLIT):]
    return train_set, test_set


def augment(img_path, cx, cy):
    # load RGB image and preprocess image as required by model
    orig = cv2.imread(img_path, 1).astype('float64')
    orig = cv2.resize(orig, (IMG_H, IMG_W))
    orig = orig[:, :, [2, 1, 0]]
    img = np.expand_dims(orig, axis=0)
    img = preprocess_input(img)

    # flip image and convert label
    x_augmented = img
    y_augmented = np.array([[cx, cy]])
    return x_augmented, y_augmented


def read_images(image_name_list, src_path):
    num_images = len(image_name_list)
    print '-- This set has {} images.'.format(num_images)

    # read images and labels
    images = []
    labels = []
    label_dict = load_labels(LABEL_FILE)
    for i, image_name in enumerate(image_name_list):
        image_path = os.path.join(src_path, image_name)
        # image = process_image(image)

        # get coordinates and transform them based on shape
        cx, cy = label_dict[image_name]
        images_flp, lbs_flp = augment(image_path, cx, cy)
        images.append(images_flp)
        labels.append(lbs_flp)

    x = np.concatenate(tuple(images), axis=0)
    y = np.concatenate(tuple(labels), axis=0)
    return x, y


def load_data(src_path):

    # get list of image names and partition into train and test sets
    file_list = os.listdir(src_path)
    train_set = [fname for fname in file_list if fname.endswith('.jpg')]
    x_train, y_train = read_images(train_set, src_path)
    print 'train:', x_train.shape, y_train.shape
    return x_train, y_train


def compute_accuracy(predictions, gtruth):
    diff = predictions - gtruth
    dist = np.sqrt(np.linalg.norm(diff, axis=1))
    num_correct = (dist <= RADIUS).sum()
    accuracy = float(num_correct) / predictions.shape[0]
    return accuracy, dist


def main():
    # load data
    X_train, Y_train = load_data(TRAIN_DIR)

    #load model
    model = load_model(MODEL_PATH)

    # check results
    train_preds = model.predict(x=X_train)
    ac, error = compute_accuracy(train_preds, Y_train)
    print 'Train accuracy:', ac


if __name__ == '__main__':
    main()
