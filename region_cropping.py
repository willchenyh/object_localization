"""
crop an image to small regions.
region size: 1/4 height, 1/5 width
"""

import cv2
import os

REGIONS_PATH = 'regions'
SRC_PATH = 'find_phone'
LABEL_FILE = 'labels.txt'
IMG_NAME = '0.jpg'
STEP_PCT = 0.25  # of small region


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


def crop_regions(src_path, img_name):

    orig = cv2.imread(os.path.join(src_path, img_name), 1).astype('float64')

    # compute region height and width
    orig_height, orig_width = orig.shape[0], orig.shape[1]
    reg_height, reg_width = orig_height / 4, orig_width / 5

    # load label dict
    label_dict = load_labels(os.path.join(src_path, LABEL_FILE))
    x_normalized, y_normalized = label_dict[img_name]
    x_pixel, y_pixel = x_normalized * orig_width, y_normalized * orig_height

    # crop regions, with half overlap. => 7 along height, 9 along width
    regions = []
    row_step_size = int(reg_height * 0.25)
    col_step_size = int(reg_width * 0.25)
    row_start_idx = range(0, orig_height-reg_height, row_step_size)
    col_start_idx = range(0, orig_width-reg_width, col_step_size)

    for row_start in row_start_idx:
        for col_start in col_start_idx:
            row_end = row_start + reg_height
            col_end = col_start + reg_width
            if row_start < y_pixel < row_end and col_start < x_pixel < col_end:
                binary = 'pos'
            else:
                binary = 'neg'
            folder = os.path.join(REGIONS_PATH, binary)
            region = orig[row_start:row_end, col_start:col_end, :]
            region_name = '{}_r{}_c{}_{}.jpg'.format(img_name, row_start, col_start, binary)
            cv2.imwrite(os.path.join(folder, region_name), region)

    return regions


def load_data(src_path):
    # get list of image names
    file_list = os.listdir(src_path)
    train_set = [fname for fname in file_list if fname.endswith('.jpg')]

    for image_name in train_set:
        # orig = orig[:, :, [2, 1, 0]]  # convert to RGB

        # crop regions
        regions = crop_regions(src_path, image_name)


        # # concat
        # img = np.expand_dims(orig, axis=0)
        # img = preprocess_input(img)


def main():
    load_data(SRC_PATH)


if __name__ == '__main__':
    main()
