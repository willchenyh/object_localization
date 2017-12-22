"""
This program creates a Convolutional Neural Network and train on data provided.

Author: Yuhan (Will) Chen
"""

import sys
import numpy as np
import os
import cv2
from models import build_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

IMG_H, IMG_W = 224, 224
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_COORDS = 2
LABEL_FILE = 'labels.txt'
MODEL_NAME = 'vgg16'
WEIGHTS_PATH = 'find_phone_{}_weights.h5'.format(MODEL_NAME)


def load_labels(file_path):
    """
    Reads labels and save them in a dictionary.
    :param file_path: label file path
    :return: dictionary of images with their labels
    """
    assert isinstance(file_path, str)

    f = open(file_path, 'rb')
    content = f.readlines()
    label_dict = {}
    for label in content:
        label_parts = label.strip().split(' ')
        label_dict[label_parts[0]] = (float(label_parts[1]), float(label_parts[2]))
    return label_dict


def augment(img_path, cx, cy):
    """
    Flips input image horizontally and vertically.
    :param img_path: image path
    :param cx: x coordinate
    :param cy: y coordinate
    :return: numpy arrays of augmented images and coordinates
    """
    assert isinstance(img_path, str)
    assert isinstance(cx, float) and 0 <= cx <= 1
    assert isinstance(cy, float) and 0 <= cy <= 1

    # load RGB image and preprocess image as required by model
    orig = cv2.imread(img_path, 1).astype('float64')
    orig = cv2.resize(orig, (IMG_H, IMG_W))
    orig = orig[:,:,[2,1,0]]
    img = np.expand_dims(orig, axis=0)
    img = preprocess_input(img)

    # flip image and convert label
    img_lr = image.flip_axis(img, 2)
    img_ud = image.flip_axis(img, 1)
    img_lrud = image.flip_axis(img_lr, 1)
    x_augmented = np.concatenate((img,img_lr,img_ud,img_lrud), axis=0)
    y_augmented = np.array([[cx, cy], [1 - cx, cy], [cx, 1 - cy], [1 - cx, 1 - cy]])
    return x_augmented, y_augmented


def load_data(src_path):
    """
    Load images and labels from the input directory.
    :param src_path: directory with images and label file
    :return: numpy arrays of images and labels
    """
    assert isinstance(src_path, str)

    # get list of image names
    file_list = os.listdir(src_path)
    train_set = [fname for fname in file_list if fname.endswith('.jpg')]

    # read images into numpy arrays
    images = []
    labels = []
    label_dict = load_labels(os.path.join(src_path,LABEL_FILE))
    for image_name in train_set:
        image_path = os.path.join(src_path, image_name)
        # get coordinates
        cx, cy = label_dict[image_name]
        # augment this image
        images_augmented, lbs_augmented = augment(image_path, cx, cy)
        images.append(images_augmented)
        labels.append(lbs_augmented)
    x = np.concatenate(tuple(images), axis=0)
    y = np.concatenate(tuple(labels), axis=0)
    return x, y


def main(argv):
    """
    Creates a Convolutional Neural Network, train on the data in input source directory, and save weights.
    :param argv: list of command line arguments
    :return: none
    """
    assert isinstance(argv, list)
    assert len(argv) == 1

    # read command line arguments
    train_dir = argv[0]
    # make model
    model = build_model(MODEL_NAME)
    # Get data
    x_train, y_train = load_data(train_dir)
    # Train model
    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1)
    # Save model weights
    model.save(WEIGHTS_PATH)


if __name__ == '__main__':
    main(sys.argv[1:])
