"""
crop an image to small regions.
region size: 1/4 height, 1/5 width
"""

import cv2
import os

REGIONS_PATH = 'regions'


def crop_regions(orig):
    # compute region height and width
    orig_height, orig_width = orig.shape[0], orig.shape[1]
    reg_height, reg_width = orig_height / 4, orig_width / 5

    # crop regions, with half overlap. => 7 along height, 9 along width
    regions = []
    for row in range(7):
        for col in range(9):
            row_start = row * reg_height / 2
            col_start = col * reg_width / 2
            region = orig[row_start:row_start+reg_height, col_start:col_start+reg_width, :]
            region_name = 'test_r{}_c{}.jpg'.format(row, col)
            cv2.imwrite(os.path.join(REGIONS_PATH,region_name), region)

    return regions


def main():
    # read image
    orig = cv2.imread('find_phone/0.jpg', 1).astype('float64')
    # orig = orig[:, :, [2, 1, 0]]  # convert to RGB

    # crop regions
    regions = crop_regions(orig)


    # # concat
    # img = np.expand_dims(orig, axis=0)
    # img = preprocess_input(img)


if __name__ == '__main__':
    main()
