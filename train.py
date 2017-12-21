"""
Author: Yuhan (Will) Chen
"""

# from keras.models import Model
# from keras.applications.vgg16 import VGG16
# from keras import optimizers
# from keras.layers import Dropout, Flatten, Dense
# from keras.utils.np_utils import to_categorical
from load_model import load_model
from keras.preprocessing import image

import numpy as np
# import glob
import os
import cv2
import random

# # specific to xception
# from keras.applications.xception import preprocess_input
# IMG_H, IMG_W, NUM_CHANNELS = 299, 299, 3

# # specific to inceptionResNetV2
# from keras.applications.inception_resnet_v2 import preprocess_input
# IMG_H, IMG_W, NUM_CHANNELS = 299, 299, 3
# MODEL_NAME = 'inception_resnet_v2'

# specific to vgg16
from keras.applications.vgg16 import preprocess_input
IMG_H, IMG_W, NUM_CHANNELS = 224, 224, 3

# MEAN_PIXEL = np.array([104., 117., 123.]).reshape((1,1,3))
TRAIN_DIR = '../find_phone'
LABEL_FILE = '../find_phone/labels.txt'
TEST_SPLIT = 0.1
# VAL_DIR = '../data/validation'
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_COORDS = 2
TASK_NAME = 'find_phone'
RADIUS = 0.05

INDEX_FILE = 'index_file.txt'
CIRCLE_DIR = '../test_augmentation'

'''
def load_model():
    # build the VGG16 network
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_H, IMG_W, NUM_CHANNELS))
    print('Model weights loaded.')
    base_out = base_model.output
    flat = Flatten()(base_out)
    hidden = Dense(4096, activation='relu')(flat)
    hidden = Dense(4096, activation='relu')(hidden)
    # hidden = Dense(256, activation='relu')(hidden)
    # drop = Dropout(0.5)(hidden)
    # hidden = Dense(32, activation='relu')(hidden)
    predictions = Dense(NUM_COORDS, activation='sigmoid')(hidden)
    model = Model(inputs=base_model.input, outputs=predictions)
    print 'Build model'

    # train only the top layers
    for layer in model.layers[:19]:
        layer.trainable = False

    # compile the model
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss='mean_squared_error', metrics=['mse'])
    print 'Compile model'
    model.summary()
    return model
'''

def process_image(image):
    # zero pad 5-pixel boundary
    temp = np.zeros((IMG_H+10, IMG_W+10, NUM_CHANNELS))
    temp[5:IMG_H+5, 5:IMG_H+5, :] = image
    # random horizontal flip
    flip = np.asarray(range(2))
    flip_choice = np.random.choice(flip)
    if flip_choice == 1:
        temp = cv2.flip(temp, 1)
    # random cropping
    crop = np.asarray(range(10))
    crop_choice = np.random.choice(crop, 2, False)  # starting pixel location
    row, col = crop_choice[0], crop_choice[1]
    new_image = temp[row:row+IMG_H, col:col+IMG_W, :]
    return new_image


def save_index(idx_list):
    f = open(INDEX_FILE, 'wb')
    for fname in idx_list:
        f.write(str(fname)+'\n')


def load_labels(file_path):
    f = open(file_path, 'rb')
    content = f.readlines()
    label_dict = {}
    for label in content:
        label_parts = label.strip().split(' ')
        label_dict[label_parts[0]] = (float(label_parts[1]), float(label_parts[2]))
    return label_dict


def partition_data(image_list):
    """
    partition sets using the image name list
    :param X:
    :param Y:
    :return: training set list, test set list (both with image names)
    """
    n_samples = len(image_list)
    # idx_list = range(n_samples)
    random.shuffle(image_list)

    # save_index(image_list)

    test_set = image_list[:int(n_samples*TEST_SPLIT)]
    train_set = image_list[int(n_samples*TEST_SPLIT):]
    return train_set, test_set


def draw_circle(image, cx, cy, image_path, code):
    # TODO: draw circle on phone and save file
    temp = image.copy()
    cx = int(cx * IMG_W)
    cy = int(cy * IMG_H)
    cv2.circle(temp, (cx, cy), 20, (0,0,255), 1)

    orig_name = image_path.split('/')[-1]
    cv2.imwrite(os.path.join(CIRCLE_DIR, orig_name+'_'+code+'.jpg'), temp)
    return


def augment(image_path, cx, cy):
    # image = cv2.imread(image_path, 1)
    # image = cv2.resize(image, (IMG_H, IMG_W))

    orig = image.load_img(image_path, target_size=(IMG_H, IMG_H))
    orig = image.img_to_array(orig)
    img = np.expand_dims(orig, axis=0)
    img = preprocess_input(img)

    # x_augmented = np.zeros((4, IMG_H, IMG_W, NUM_CHANNELS))
    # orig = image - MEAN_PIXEL
    # x_augmented[0,:,:,:] = img
    y_augmented = np.array([[cx, cy], [1-cx, cy], [cx, 1-cy], [1-cx, 1-cy]])

    img_lr = image.flip_axis(img, 2)
    img_ud = image.flip_axis(img, 1)
    img_lrud = image.flip_axis(img_lr, 1)
    x_augmented = np.concatenate((img,img_lr,img_ud,img_lrud), axis=0)

    # flip_codes = [1, 0, -1]  # hor, ver, both
    # for i,fc in enumerate(flip_codes):
    #     flipped = cv2.flip(image, fc)
    #     flipped_norm = flipped - MEAN_PIXEL
    #     augmented[i+1,:,:,:] = flipped_norm
    #
    #     # TODO: to test and save images with a drawn label
    #     draw_circle(flipped_norm, y[i+1, 0], y[i+1, 1], image_path, str(fc))

    # TODO: to test and save images with a drawn label
    draw_circle(orig, cx, cy, image_path, '')
    # draw_circle(img_lr, cx, cy, image_path, '')

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

        # get coordinates
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
    image_name_list = [fname for fname in file_list if fname.endswith('.jpg')]
    train_set, test_set = partition_data(image_name_list)

    # read images into numpy arrays
    x_train, y_train = read_images(train_set, src_path)
    print 'train:', x_train.shape, y_train.shape
    x_test, y_test = read_images(test_set, src_path)
    print 'test:', x_test.shape, y_test.shape
    return x_train, y_train, x_test, y_test


def compute_accuracy(predictions, gtruth):
    diff = predictions - gtruth
    dist = np.linalg.norm(diff, axis=1)
    num_correct = (dist <= RADIUS).sum()
    accuracy = float(num_correct) / predictions.shape[0]
    return accuracy, dist


def visualize_test(x_test, y_preds):
    for i in range(x_test.shape[0]):
        img = x_test[i,:,:,:]
        # cx_lb, cy_lb = y_test[i,0], y_test[i,1]
        cx_pred, cy_pred = y_preds[i,0], y_preds[i,1]
        draw_circle(img, cx_pred, cy_pred, '/test_'+str(i), '')
    return


def main():
    # make model
    model = load_model(MODEL_NAME)
    print MODEL_NAME, 'created\n'
    # Get data
    print 'Load data:'
    x_train, y_train, x_test, y_test = load_data(TRAIN_DIR)
    # print 'Load val data:'
    # X_val, Y_val = load_data(VAL_DIR)
    # Train model
    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1)
    print '\n'
    # Save model weights
    model.save('../vgg16_{}_weights.h5'.format(TASK_NAME))
    print 'model weights saved.'

    # use model on test set
    results = model.evaluate(x_test, y_test)
    print 'Test results:'
    print results[0]

    # check results
    from sklearn.metrics import mean_squared_error as MSE

    train_preds = model.predict(x=x_train)
    ac, error = compute_accuracy(train_preds, y_train)
    print 'Train accuracy:', ac
    mse_train = MSE(y_train, train_preds)
    print 'Train MSE:', mse_train

    test_preds = model.predict(x=x_test)
    ac, error = compute_accuracy(test_preds, y_test)
    print 'Test accuracy:', ac
    mse_test = MSE(y_test, test_preds)
    print 'Test MSE:', mse_test
    visualize_test(x_test, test_preds)
    print error


if __name__ == '__main__':
    main()
