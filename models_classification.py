from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras import optimizers
from keras.layers import Flatten, Dense, Dropout

NUM_CLASSES = 2


def build_model(model_name):
    """
    Build a model based on input model name
    :param model_name: name of model
    :return: keras model instance
    """
    assert isinstance(model_name, str)
    assert model_name in ['vgg16', 'xception']

    if model_name == 'vgg16':
        return build_vgg16()
    # if model_name == 'xception':
    #     return build_xception()


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
    # x = Dropout(0.5)(x)
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
