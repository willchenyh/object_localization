"""
This program predicts phone locations in new images, using a trained model.

Author: Yuhan (Will) Chen
"""

import sys
import os
import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

IMG_H, IMG_W = 224, 224
WEIGHTS_PATH = 'find_phone_classifier_weights.h5'
# LABEL_FILE = 'labels.txt'
STEP_PCT = 0.20  # of small region
REGION_PCT = 1.0 / 6.0  # of sides of original image
# NUM_RG_PER_IMG = 728


# def read_img(img_path):
#     """
#     Read and process image as required by model.
#     :param img_path: path of image
#     :return: numpy array of image
#     """
#     assert isinstance(img_path, str)
#
#     orig = cv2.imread(img_path, 1).astype('float64')
#     orig = cv2.resize(orig, (IMG_H, IMG_W))
#     img = orig[:, :, [2, 1, 0]]
#     # img = np.expand_dims(orig, axis=0)
#     img = preprocess_input(img)
#     return img


# def load_labels(file_path):
#     """
#     Reads labels and save them in a dictionary.
#     :param file_path: label file path
#     :return: dictionary of images with their labels
#     """
#     assert isinstance(file_path, str)
#
#     f = open(file_path, 'rb')
#     content = f.readlines()
#     label_dict = {}
#     for label in content:
#         label_parts = label.strip().split(' ')
#         label_dict[label_parts[0]] = (float(label_parts[1]), float(label_parts[2]))
#     return label_dict


def predict(img_path, model):

    orig = cv2.imread(img_path, 1).astype('float64')
    orig = orig[:, :, [2, 1, 0]]
    orig = preprocess_input(orig)

    # compute region height and width
    orig_height, orig_width = orig.shape[0], orig.shape[1]
    reg_height, reg_width = int(orig_height * REGION_PCT), int(orig_width * REGION_PCT)
    # print img_name
    # print orig_height, orig_width
    # print reg_height, reg_width

    # load label dict
    # label_dict = load_labels(os.path.join(src_path, LABEL_FILE))
    # x_normalized, y_normalized = label_dict[img_name]
    # x_pixel, y_pixel = x_normalized * orig_width, y_normalized * orig_height

    # crop regions, with half overlap. => 7 along height, 9 along width
    # regions = []
    row_step_size = int(reg_height * STEP_PCT)
    col_step_size = int(reg_width * STEP_PCT)
    row_start_idx = range(0, orig_height - reg_height, row_step_size)
    col_start_idx = range(0, orig_width - reg_width, col_step_size)

    # pos_regions = np.array([[-1, -1]])  # two starting coordinates

    num_regions = len(row_start_idx) * len(col_start_idx)
    regions = np.zeros((num_regions, IMG_H, IMG_W, 3))
    # region_labels = np.zeros((num_regions, 1))
    for i, row_start in enumerate(row_start_idx):
        for j, col_start in enumerate(col_start_idx):
            row_end = row_start + reg_height
            col_end = col_start + reg_width

            region = orig[row_start:row_end, col_start:col_end, :]
            region = cv2.resize(region, (IMG_H, IMG_W))
            region = np.expand_dims(region, axis=0)

            regions[i * len(col_start_idx) + j, :, :, :] = region

    scores = model.predict(x=regions)
    sorted_idx = np.argsort(scores)  # sort index by row
    labels = sorted_idx[:, -1]  # 1d array

    # print 'scores'
    # print scores
    # print 'sorted idx'
    # print sorted_idx
    # print labels

    pos_idx = np.argwhere(labels == 1)  # 2d array nx1

    # print 'pos idx'
    # print pos_idx

    row_nums = np.divide(pos_idx, len(col_start_idx))
    row_idx = row_nums * row_step_size
    col_nums = np.remainder(pos_idx, len(col_start_idx))
    col_idx = col_nums * col_step_size

    # rnc = np.concatenate((row_idx, col_idx), axis=1)
    # pos_mean = np.mean(rnc, axis=0)

    pos = np.concatenate((col_idx, row_idx), axis=1)
    pos_mean = np.mean(pos, axis=0)
    center_coord = pos_mean + 0.5 * np.array([[reg_width, reg_height]])
    center_normal = np.divide(center_coord, np.array([[orig_width, orig_height]]))


    # print pos_mean

    # folder = os.path.join(REGIONS_PATH, binary)
    # regions[i*len(col_start_idx)+j, :, :, :] = region

    # region_name = '{}_r{}_c{}_{}.jpg'.format(img_name, row_start, col_start, binary)
    # cv2.imwrite(os.path.join(folder, region_name), region)

    # pos_regions = pos_regions[1:, :]
    # pos_mean = np.mean(pos_regions, axis=0)

    # center_row = pos_mean[0,0] + 0.5 * reg_height
    # center_col = pos_mean[0,1] + 0.5 * reg_width
    # center_coord = pos_mean + 0.5 * np.array([[reg_height, reg_width]])
    # center_normal = np.divide(center_coord, np.array([[orig_height, orig_width]]))

    # dist = np.linalg.norm(center_normal - np.array([[y_normalized, x_normalized]]))

    # print 'center', center_normal, 'truth', (y_normalized, x_normalized)

    # correct = 0
    # if dist <= 0.05:
    #     # print 'yayyyyyy!!!!!!!!!!!!!!!!!'
    #     correct = 1
    # else:
    #     print dist
    # print img_name, correct
    # print 'dist', dist, '     ', correct
    # print
    return np.around(center_normal, decimals=4)


# def test_on_all(src_path, model):
#     # file_list = os.listdir(src_path)
#     # img_name_list = sorted([fname for fname in file_list if fname.endswith('.jpg')])
#
#     # get set partitions
#     f = open('random_list.txt', 'rb')
#     img_name_list = f.readlines()
#     img_name_list = [img_name.strip() for img_name in img_name_list]
#
#     # split input list and return train and val img names
#     num_samples = len(img_name_list)
#     # num_train = int(num_samples * (1 - val_ratio))
#     # train_imgs = img_name_list[:num_train]
#     # val_imgs = img_name_list[num_train:]
#
#
#     idx = 1
#     print 'Using test index', idx
#
#     val_ratio = 0.1
#     test_ratio = 0.2
#
#     num_test = int(num_samples * test_ratio)
#     # for idx in range(5):
#     if idx == 4:
#         test_imgs = img_name_list[idx * num_test:]
#         train_val_imgs = img_name_list[:idx * num_test]
#     else:
#         test_imgs = img_name_list[idx * num_test:(idx + 1) * num_test]
#         train_val_imgs = img_name_list[:idx * num_test] + img_name_list[(idx + 1) * num_test:]
#         # break
#
#     num_train = int(len(train_val_imgs) * (1 - val_ratio))
#
#     train_imgs = train_val_imgs[:num_train]
#     val_imgs = train_val_imgs[num_train:]
#
#     sets_list = [train_imgs, val_imgs, test_imgs]
#
#     # print sets_list
#     for i, dataset in enumerate(sets_list):
#         correct = 0
#         # # TODO=================================================================================================
#         # dataset = ['0.jpg','1.jpg']
#         # # TODO=================================================================================================
#         for img_name in dataset:
#             correct += crop_regions(src_path, img_name, model)
#         accuracy = correct / float(len(dataset))
#         print '========================='
#         print 'set idx:', i
#         print 'total accuracy', accuracy, '\n'


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
    # load model
    model = load_model(WEIGHTS_PATH)
    # crop_regions(path, name, model)
    coords = predict(img_path, model)

    # predict and print coordinates
    # coords = model.predict(x=img)
    print coords[0,0], coords[0,1]


if __name__ == '__main__':
    main(sys.argv[1:])
