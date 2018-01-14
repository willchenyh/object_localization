"""
This program creates a Convolutional Neural Network and train on data provided.

Author: Yuhan (Will) Chen
"""

import sys
import numpy as np
import os
import cv2
from classifier import build_vgg16
# from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import to_categorical

IMG_H, IMG_W = 224, 224
NUM_EPOCHS = 3
BATCH_SIZE = 32
NUM_COORDS = 2
LABEL_FILE = 'labels.txt'
WEIGHTS_PATH = 'find_phone_classifier_weights.h5'
STEP_PCT = 0.20  # of small region
REGION_PCT = 1.0 / 6.0  # of sides of original image
NUM_RG_PER_IMG = 728


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
    img_name_list = []
    for label in content:
        label_parts = label.strip().split(' ')
        img_name_list.append(label_parts[0])
        label_dict[label_parts[0]] = (float(label_parts[1]), float(label_parts[2]))
    return img_name_list, label_dict


# def augment(img_path, cx, cy):
#     """
#     Flips input image horizontally and vertically.
#     :param img_path: image path
#     :param cx: x coordinate
#     :param cy: y coordinate
#     :return: numpy arrays of augmented images and coordinates
#     """
#     assert isinstance(img_path, str)
#     assert isinstance(cx, float) and 0 <= cx <= 1
#     assert isinstance(cy, float) and 0 <= cy <= 1
#
#     # load RGB image and preprocess image as required by model
#     orig = cv2.imread(img_path, 1).astype('float64')
#     orig = cv2.resize(orig, (IMG_H, IMG_W))
#     orig = orig[:,:,[2,1,0]]
#     img = np.expand_dims(orig, axis=0)
#     img = preprocess_input(img)
#
#     # flip image and convert label
#     img_lr = image.flip_axis(img, 2)
#     img_ud = image.flip_axis(img, 1)
#     img_lrud = image.flip_axis(img_lr, 1)
#     x_augmented = np.concatenate((img,img_lr,img_ud,img_lrud), axis=0)
#     y_augmented = np.array([[cx, cy], [1 - cx, cy], [cx, 1 - cy], [1 - cx, 1 - cy]])
#     return x_augmented, y_augmented


def crop_regions(src_path, img_name, label):

    orig = cv2.imread(os.path.join(src_path, img_name), 1).astype('float64')
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
    (x_normalized, y_normalized) = label
    x_pixel, y_pixel = x_normalized * orig_width, y_normalized * orig_height

    # crop regions, with half overlap. => 7 along height, 9 along width
    # regions = []
    row_step_size = int(reg_height * STEP_PCT)
    col_step_size = int(reg_width * STEP_PCT)
    row_start_idx = range(0, orig_height-reg_height, row_step_size)
    col_start_idx = range(0, orig_width-reg_width, col_step_size)

    # pos_regions = np.array([[-1, -1]])  # two starting coordinates

    # save cropped samples and their binary labels to numpy arrays
    num_regions = len(row_start_idx) * len(col_start_idx)
    regions = np.zeros((num_regions, IMG_H, IMG_W, 3))
    region_labels = np.zeros((num_regions, 1))
    # crop samples
    for i, row_start in enumerate(row_start_idx):
        for j, col_start in enumerate(col_start_idx):
            row_end = row_start + reg_height
            col_end = col_start + reg_width
            if row_start < y_pixel < row_end and col_start < x_pixel < col_end:
                # binary = 'pos'
                region_labels[i * len(col_start_idx) + j, :] = 1
                # add coords to array
                # coords = np.array([[row_start, col_start]])
                # pos_regions = np.concatenate((pos_regions, coords), axis=0)
                # print pos_regions.shape
            else:
                # binary = 'neg'
                region_labels[i * len(col_start_idx) + j, :] = 0
            # folder = os.path.join(REGIONS_PATH, binary)
            region = orig[row_start:row_end, col_start:col_end, :]
            region = cv2.resize(region, (IMG_H, IMG_W))
            regions[i*len(col_start_idx)+j, :, :, :] = region

            # region_name = '{}_r{}_c{}_{}.jpg'.format(img_name, row_start, col_start, binary)
            # cv2.imwrite(os.path.join(folder, region_name), region)
    # pos_regions = pos_regions[1:, :]
    # pos_mean = np.mean(pos_regions, axis=0)
    # center_row = pos_mean[0,0] + 0.5 * reg_height
    # center_col = pos_mean[0,1] + 0.5 * reg_width
    # center_coord = pos_mean + 0.5 * np.array([[reg_height, reg_width]])
    # print center_coord
    # center_normal = np.divide(center_coord, np.array([[orig_height, orig_width]]))
    # dist = np.linalg.norm(center_normal - np.array([[y_normalized, x_normalized]]))
    # if dist > 0.04:
    #     print dist
    return regions, region_labels


# def load_data(src_path):
#     """
#     Load images and labels from the input directory.
#     :param src_path: directory with images and label file
#     :return: numpy arrays of images and labels
#     """
#     assert isinstance(src_path, str)
#
#     # get list of image names
#     file_list = os.listdir(src_path)
#     train_set = [fname for fname in file_list if fname.endswith('.jpg')]
#
#     # read images into numpy arrays
#     images = []
#     labels = []
#     label_dict = load_labels(os.path.join(src_path,LABEL_FILE))
#
#     # TODO use some for testing
#     # train_set = train_set[:10]
#     for image_name in train_set:
#         # image_path = os.path.join(src_path, image_name)
#         # get coordinates
#         cx, cy = label_dict[image_name]
#         # # augment this image
#         # images_augmented, lbs_augmented = augment(image_path, cx, cy)
#         # images.append(images_augmented)
#         # labels.append(lbs_augmented)
#
#         # crop regions and get labels
#         regions, rg_labels = crop_regions(src_path, image_name)
#         images.append(regions)
#         labels.append(rg_labels)
#
#     x = np.concatenate(tuple(images), axis=0)
#     y = np.concatenate(tuple(labels), axis=0)
#     y = to_categorical(y, 2)
#     return x, y


def data_partition(src_path, img_name_list, val_ratio=0.1):
    # get list of image names
    # file_list = os.listdir(src_path)
    # img_name_list = [fname for fname in file_list if fname.endswith('.jpg')]
    # f = open('random_list.txt', 'rb')
    # img_name_list = f.readlines()
    # img_name_list = [img_name.strip() for img_name in img_name_list]

    # split input list and return train and val img names
    num_samples = len(img_name_list)
    # num_train = int(num_samples * (1 - val_ratio))
    # train_imgs = img_name_list[:num_train]
    # val_imgs = img_name_list[num_train:]


    # idx = 1
    # print 'train using test idx', idx
    #
    # num_test = int(num_samples * test_ratio)
    # # for idx in range(5):
    # if idx == 4:
    #     test_imgs = img_name_list[idx * num_test:]
    #     train_val_imgs = img_name_list[:idx*num_test]
    # else:
    #     test_imgs = img_name_list[idx*num_test:(idx+1)*num_test]
    #     train_val_imgs = img_name_list[:idx*num_test] + img_name_list[(idx+1)*num_test:]
    #     # break

    num_train = int(num_samples * (1 - val_ratio))

    train_imgs = img_name_list[:num_train]
    val_imgs = img_name_list[num_train:]

    # num_train = int(num_samples * (1 - val_ratio - test_ratio))
    # num_val = int(num_samples * val_ratio)
    # train_imgs = img_name_list[:num_train]
    # val_imgs = img_name_list[num_train:num_train+num_val]
    # test_imgs = img_name_list[num_train+num_val:]

    # write sets to a file for testing
    # f = open('sets_partition.txt', 'wb')
    # for img in train_imgs:
    #     f.write(img+' 0\n')
    # for img in val_imgs:
    #     f.write(img+' 1\n')
    # for img in test_imgs:
    #     f.write(img+' 2\n')

    return train_imgs, val_imgs


def data_gen(src_path, img_name_list, label_dict):

    while True:
        for img_name in img_name_list:
            regions, rg_labels = crop_regions(src_path, img_name, label_dict[img_name])
            rg_labels = to_categorical(rg_labels, 2)
            num_samples = regions.shape[0]
            steps = range(0, num_samples, BATCH_SIZE)
            # print num_samples
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

    # read command line arguments
    train_dir = argv[0]
    # make model
    model = build_vgg16()
    # get labels and data lists
    img_name_list, label_dict = load_labels(os.path.join(train_dir, LABEL_FILE))
    train_list, val_list = data_partition(train_dir, img_name_list)
    # create data generator for training
    train_gen = data_gen(train_dir, train_list, label_dict)
    val_gen = data_gen(train_dir, val_list, label_dict)

    # Train model
    # model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1)
    num_train = len(train_list) * NUM_RG_PER_IMG
    num_val = len(val_list) * NUM_RG_PER_IMG

    # print 
    # print num_samples // BATCH_SIZE

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=num_train // BATCH_SIZE,
                        # steps_per_epoch=20,
                        epochs=NUM_EPOCHS,
                        # epochs=3,
                        validation_data=val_gen,
                        # validation_data=train_gen,
                        validation_steps=num_val // BATCH_SIZE,
                        # validation_steps=20,
                        )

    # (x, y) = next(train_gen)
    # print model.evaluate(x,y)
    # (x, y) = next(val_gen)
    # print model.evaluate(x,y)


    # Save model weights
    model.save(WEIGHTS_PATH)


if __name__ == '__main__':
    main(sys.argv[1:])
