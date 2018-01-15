"""
This program creates a Convolutional Neural Network and train on data provided.

Author: Yuhan (Will) Chen
"""

import sys
import os
import random
import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.layers import Flatten, Dense, Dropout


IMG_H, IMG_W = 224, 224
NUM_EPOCHS = 3
BATCH_SIZE = 32
NUM_CLASSES = 2
LABEL_FILE = 'labels.txt'
WEIGHTS_PATH = 'find_phone_classifier_weights.h5'
STEP_PCT = 0.20  # of small region
REGION_PCT = 1.0 / 6.0  # of sides of original image
NUM_RG_PER_IMG = 728


def build_vgg16():
    """
    Build the VGG16 network
    :return: keras model instance
    """

    # build layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_out = base_model.output
    flat = Flatten()(base_out)
    x = Dense(4096, activation='relu')(flat)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_labels(file_path):
    """
    Reads labels and save them in a dictionary.
    :param file_path: label file path
    :return: a list of image names, and a dictionary of images with their labels
    """
    assert isinstance(file_path, str)

    f = open(file_path, 'rb')
    content = f.readlines()
    label_dict = {}
    img_name_list = []
    for label in content:
        label_parts = label.strip().split(' ')
        img_name_list.append(label_parts[0])
        label_dict[label_parts[0]] = (float(label_parts[1]), float(label_parts[2]))
    random.shuffle(img_name_list)
    return img_name_list, label_dict


def crop_regions(src_path, img_name, label):
    """
    Crop region samples and assign binary labels
    :param src_path: path of data
    :param img_name: image name at the path
    :param label: normalized coordinates of phone
    :return: region samples and binary labels in numpy arrays
    """
    assert isinstance(src_path, str)
    assert isinstance(img_name, str)
    assert isinstance(label, tuple) and len(label) == 2

    # read image and pre-process as required
    orig = cv2.imread(os.path.join(src_path, img_name), 1).astype('float64')
    orig = orig[:, :, [2, 1, 0]]
    orig = preprocess_input(orig)

    # compute region height and width
    orig_height, orig_width = orig.shape[0], orig.shape[1]
    reg_height, reg_width = int(orig_height * REGION_PCT), int(orig_width * REGION_PCT)

    # load label
    (x_normalized, y_normalized) = label
    x_pixel, y_pixel = x_normalized * orig_width, y_normalized * orig_height

    # crop regions, save them and their binary labels to numpy arrays
    row_step_size = int(reg_height * STEP_PCT)
    col_step_size = int(reg_width * STEP_PCT)
    row_start_idx = range(0, orig_height-reg_height, row_step_size)
    col_start_idx = range(0, orig_width-reg_width, col_step_size)
    num_regions = len(row_start_idx) * len(col_start_idx)
    regions = np.zeros((num_regions, IMG_H, IMG_W, 3))
    region_labels = np.zeros((num_regions, 1))
    for i, row_start in enumerate(row_start_idx):
        for j, col_start in enumerate(col_start_idx):
            row_end = row_start + reg_height
            col_end = col_start + reg_width
            if row_start < y_pixel < row_end and col_start < x_pixel < col_end:
                region_labels[i * len(col_start_idx) + j, :] = 1
            else:
                region_labels[i * len(col_start_idx) + j, :] = 0
            region = orig[row_start:row_end, col_start:col_end, :]
            region = cv2.resize(region, (IMG_H, IMG_W))
            regions[i*len(col_start_idx)+j, :, :, :] = region
    return regions, region_labels


def data_partition(src_path, img_name_list, val_ratio=0.1):
    """
    Split input list and return train and val img names
    :param src_path: path of data
    :param img_name_list: list of image names at the path
    :param val_ratio: ratio of validation set
    :return: lists of training and validation images
    """
    assert isinstance(src_path, str)
    assert isinstance(img_name_list, list) and len(img_name_list) > 0
    assert 0. <= val_ratio <= 1.

    num_samples = len(img_name_list)
    num_train = int(num_samples * (1 - val_ratio))
    train_imgs = img_name_list[:num_train]
    val_imgs = img_name_list[num_train:]
    return train_imgs, val_imgs


def data_gen(src_path, img_name_list, label_dict):
    """
    A generator that yields data batches for training and validation
    :param src_path: path of data
    :param img_name_list: list of image names at the path
    :param label_dict: dictionary of labels where keys are image names and values are coordinates
    :return: batch of data in numpy array
    """
    assert isinstance(src_path, str)
    assert isinstance(img_name_list, list) and len(img_name_list) > 0
    assert isinstance(label_dict, dict) and len(label_dict) > 0

    while True:
        # for each image, extract region samples and yield batches
        for img_name in img_name_list:
            regions, rg_labels = crop_regions(src_path, img_name, label_dict[img_name])
            rg_labels = to_categorical(rg_labels, 2)
            num_samples = regions.shape[0]
            steps = range(0, num_samples, BATCH_SIZE)
            for idx in steps:
                yield (regions[idx:idx+BATCH_SIZE, :, :, :], rg_labels[idx:idx+BATCH_SIZE, :])


def main(argv):
    """
    Creates a Convolutional Neural Network, train on the data in input source directory, and save weights.
    :param argv: list of command line arguments
    :return: none
    """
    assert isinstance(argv, list)
    assert len(argv) == 1

    # get labels and data lists
    train_dir = argv[0]
    img_name_list, label_dict = load_labels(os.path.join(train_dir, LABEL_FILE))

    #TODO========================================================================testing purpose
    print 'testing images:'
    print img_name_list[:8]
    img_name_list = img_name_list[8:]
    #TODO========================================================================testing purpose

    train_list, val_list = data_partition(train_dir, img_name_list)

    # create data generator for training
    train_gen = data_gen(train_dir, train_list, label_dict)
    val_gen = data_gen(train_dir, val_list, label_dict)

    # make model
    model = build_vgg16()

    # Train model
    num_train = len(train_list) * NUM_RG_PER_IMG
    num_val = len(val_list) * NUM_RG_PER_IMG
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=num_train // BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_data=val_gen,
                        validation_steps=num_val // BATCH_SIZE,
                        )

    # Save model weights
    model.save(WEIGHTS_PATH)


if __name__ == '__main__':
    main(sys.argv[1:])
