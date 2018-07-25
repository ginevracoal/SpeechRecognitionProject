import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers import RepeatVector, UpSampling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16

import numpy as np

def build_simple(input_shape, n_output):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_output, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model


def build_tfclone(input_shape, n_output):
    model = Sequential()

    model.add(Conv2D(64, (20, 8), activation='relu', input_shape=input_shape))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (10, 4), activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(n_output, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

    return model


def build_trlrn_vgg(input_shape, n_output):
    pretrained_net = MobileNet(include_top=False)

    for layer in pretrained_net.layers[:-4]:
        layer.trainable = False

    model = Sequential()

    model.add(Conv2D(3, (2, 2), input_shape=input_shape))
    model.add(UpSampling2D(size=(4, 4)))

    model.add(pretrained_net)

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_output, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

    return model


def build_trlrn_resnet(input_shape, n_output):
    pretrained_net = ResNet50(include_top=False)

    for layer in pretrained_net.layers[:-4]:
        layer.trainable = False

    model = Sequential()

    model.add(Conv2D(3, (2, 2), input_shape=input_shape))
    model.add(UpSampling2D(size=(4, 4)))

    model.add(pretrained_net)

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_output, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

    return model


def build_trlrn_mobilenet(input_shape, n_output):
    pretrained_net = MobileNet(include_top=False)

    for layer in pretrained_net.layers[:-4]:
        layer.trainable = False

    model = Sequential()

    model.add(Conv2D(3, (2, 2), input_shape=input_shape))
    model.add(UpSampling2D(size=(4, 4)))

    model.add(pretrained_net)

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_output, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

    return model
