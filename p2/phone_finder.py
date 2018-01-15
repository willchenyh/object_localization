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
WEIGHTS_PATH = 'find_phone_classifier_weights.h5'
STEP_PCT = 0.20  # of small region
REGION_PCT = 1.0 / 6.0  # of sides of original image


def predict(img_path, model):
    """
    Predict if a region sample contains phone object and compute coordinates
    :param img_path: image path
    :param model: trained model
    :return: predicted coordinates
    """
    assert isinstance(img_path, str)

    # read image and pre-process as required
    orig = cv2.imread(img_path, 1).astype('float64')
    orig = orig[:, :, [2, 1, 0]]
    orig = preprocess_input(orig)

    # compute region height and width
    orig_height, orig_width = orig.shape[0], orig.shape[1]
    reg_height, reg_width = int(orig_height * REGION_PCT), int(orig_width * REGION_PCT)

    # crop regions
    row_step_size = int(reg_height * STEP_PCT)
    col_step_size = int(reg_width * STEP_PCT)
    row_start_idx = range(0, orig_height - reg_height, row_step_size)
    col_start_idx = range(0, orig_width - reg_width, col_step_size)
    num_regions = len(row_start_idx) * len(col_start_idx)
    regions = np.zeros((num_regions, IMG_H, IMG_W, 3))
    for i, row_start in enumerate(row_start_idx):
        for j, col_start in enumerate(col_start_idx):
            row_end = row_start + reg_height
            col_end = col_start + reg_width

            region = orig[row_start:row_end, col_start:col_end, :]
            region = cv2.resize(region, (IMG_H, IMG_W))
            region = np.expand_dims(region, axis=0)

            regions[i * len(col_start_idx) + j, :, :, :] = region

    # classify regions and note the positive regions
    scores = model.predict(x=regions)
    sorted_idx = np.argsort(scores)  # sort index by row
    labels = sorted_idx[:, -1]
    pos_idx = np.argwhere(labels == 1)

    # compute coordinates of phone object
    row_nums = np.divide(pos_idx, len(col_start_idx))
    row_idx = row_nums * row_step_size
    col_nums = np.remainder(pos_idx, len(col_start_idx))
    col_idx = col_nums * col_step_size
    pos = np.concatenate((col_idx, row_idx), axis=1)
    pos_mean = np.mean(pos, axis=0)
    center_coord = pos_mean + 0.5 * np.array([[reg_width, reg_height]])
    center_normal = np.divide(center_coord, np.array([[orig_width, orig_height]]))
    return np.around(center_normal, decimals=4)


def main(argv):
    """
    Load a trained model, and predict phone coordinates.
    :param argv: list of command line arguments
    :return: none
    """
    assert isinstance(argv, list)
    assert len(argv) == 1

    # read image path
    img_path = argv[0]

    # load model
    model = load_model(WEIGHTS_PATH)

    # predict and print coordinates
    coords = predict(img_path, model)
    print coords[0, 0], coords[0, 1]


if __name__ == '__main__':
    main(sys.argv[1:])
