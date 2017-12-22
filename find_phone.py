"""
This program predicts phone locations in new images, using a trained model.

Author: Yuhan (Will) Chen
"""

import sys
import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

IMG_H, IMG_W = 224, 224
MODEL_NAME = 'vgg16'
WEIGHTS_PATH = 'find_phone_{}_weights.h5'.format(MODEL_NAME)


def read_img(img_path):
    """
    Read and process image as required by model.
    :param img_path: path of image
    :return: numpy array of image
    """
    assert isinstance(img_path, str)

    orig = cv2.imread(img_path, 1).astype('float64')
    orig = cv2.resize(orig, (IMG_H, IMG_W))
    orig = orig[:, :, [2, 1, 0]]
    img = np.expand_dims(orig, axis=0)
    img = preprocess_input(img)
    return img


def main(argv):
    """
    Load a trained model, and predict phone coordinates.
    :param argv: list of command line arguments
    :return: none
    """
    assert isinstance(argv, list)
    assert len(argv) == 1

    # read command line arguments
    img_path = argv[0]
    # load image
    img = read_img(img_path)
    # load model
    model = load_model(WEIGHTS_PATH)
    # predict and print coordinates
    coords = model.predict(x=img)
    print coords[0,0], coords[0,1]


if __name__ == '__main__':
    main(sys.argv[1:])
