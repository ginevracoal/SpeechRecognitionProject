from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

def build_simple(input_shape, n_output):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


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
        optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model


def build_trlrn(input_shape, n_output):
    raise NotImplementedError
