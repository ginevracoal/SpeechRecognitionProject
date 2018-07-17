import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

def model_1():
    model = Sequential()
    # 32 2x2 neurons -> 32 19x10 objects
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    # 48 2x2 neurons -> 48 18x9 objects
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    # 120 2x2 neurons -> 120 17x8 objects
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    # 2x2 pooling (stride=2) -> 1 9x4 object
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly setting 25% of values to 0 (to prevent overfitting) -> same shape as before
    model.add(Dropout(0.25))
    # flatten to a single vector of dimension 120x9x4 = 4320
    model.add(Flatten())
    # 128 neurons of maximum dimension (4320)
    model.add(Dense(128, activation='relu'))
    # drop 25% to 0
    model.add(Dropout(0.25))
    # 64 neurons, same output
    model.add(Dense(64, activation='relu'))
    # drop 40% to 0
    model.add(Dropout(0.4))
    # 35 neurons for final classification
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def model_2():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(2,2), activation='relu', input_shape=(20, 11, 1)))
    model.add(Conv2D(64, kernel_size=(2,2), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))    
    model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(35, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model

def model_3():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(2,2), activation='relu', input_shape=(20, 11, 1)))
    model.add(Conv2D(64, kernel_size=(2,2), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))    
    model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(35, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model


def model_4():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(2,2), activation='relu', input_shape=(20, 11, 1)))
    model.add(Conv2D(64, kernel_size=(2,2), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))
    model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

#     model.add(Conv2D(256, kernel_size=(2,2), activation='relu'))
#     model.add(Conv2D(256, kernel_size=(2,2), activation='relu'))
#     model.add(Conv2D(256, kernel_size=(2,2), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))#, strides=(2,2)))

#     model.add(Conv2D(512, kernel_size=(2,2), activation='relu'))
#     model.add(Conv2D(512, kernel_size=(2,2), activation='relu'))
#     model.add(Conv2D(512, kernel_size=(2,2), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))#, strides=(2,2)))

#     model.add(Conv2D(512, kernel_size=(2,2), activation='relu'))
#     model.add(Conv2D(512, kernel_size=(2,2), activation='relu'))
#     model.add(Conv2D(512, kernel_size=(2,2), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))#, strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(35, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model