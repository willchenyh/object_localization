from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

NUM_COORDS = 2


def load_model(model_name):
    if model_name == 'vgg16':
        return load_vgg16()
    if model_name == 'xception':
        return load_xception()


def load_vgg16():
    # build the VGG16 network
    IMG_H, IMG_W, NUM_CHANNELS = 224, 224, 3
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
    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9), loss='mean_squared_error', metrics=['mse'])
    print 'Compile model'
    model.summary()
    return model


def load_xception():
    IMG_H, IMG_W, NUM_CHANNELS = 299, 299, 3
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(IMG_H,IMG_W,NUM_CHANNELS))
    print('Model weights loaded.')
    base_out = base_model.output
    x = GlobalAveragePooling2D()(base_out)
    x = Dense(2048, activation='sigmoid')(x)
    predictions = Dense(NUM_COORDS, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print 'Build model'

    # train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss='mean_squared_error', metrics=['mse'])
    print 'Compile model'
    model.summary()
    return model