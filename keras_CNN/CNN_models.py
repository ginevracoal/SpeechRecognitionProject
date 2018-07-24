from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import json

# # Feature dimension
feature_dim_1 = 32
feature_dim_2 = 32
channel = 1
epochs = 50
batch_size = 100
kernel_size = (3,3)
verbose = 1
num_classes = 35 # 35 without background noise
input_shape=(feature_dim_1, feature_dim_2, channel)

path = "/galileo/home/userexternal/gcarbone/group/keras_CNN/trained_models/"

def save_model(trained_model, model_name):
    ## save the history as a dictionary for following plots
    history_dict=trained_model.history.history
    json.dump(history_dict, open(path + model_name + '_32_history.json','w'))

    ## serialize model to JSON
    model_json = trained_model.to_json()
    with open(path + model_name + '_32.json', "w") as json_file:
        json_file.write(model_json)
    
    ## serialize weights to HDF5 (but this only saves the weights!!!)
    trained_model.save_weights(path + model_name + '_32.h5')

    print("Saved model to disk")

def load_model(model_name):
    ## load the history
    history = open(path + model_name + '_history.json', 'r')
    history_dict = json.load(history)
    history.close()

    ## load json and create model
    json_file = open(path + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    ## load weights into new model
    loaded_model.load_weights(path + model_name + '.h5')

    print("Loaded model from disk")

    return(loaded_model, history_dict)


def plot_accuracy(history):
    
    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_loss(history):
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# this is the example from here: 
# https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b
def model_1():
    model = Sequential()
    # 32 2x2 neurons -> 32 19x10 objects
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    # 48 2x2 neurons -> 48 18x9 objects
    model.add(Conv2D(48, kernel_size=kernel_size, activation='relu'))
    # 120 2x2 neurons -> 120 17x8 objects
    model.add(Conv2D(120, kernel_size=kernel_size, activation='relu'))
    # 2x2 pooling (stride=2) -> 1 9x4 object
    model.add(MaxPooling2D(pool_size=(4, 4)))
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

# this one is inspired to vgg16
def model_2():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))    

    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu'))    
    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu'))

    # one max pooling is enough!
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

    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))    
    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu'))    
    model.add(Conv2D(256, kernel_size=kernel_size, activation='relu'))    

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(35, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model



def model_4():

    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))   
    model.add(Dropout(0.25))
 
    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu'))    
    # model.add(Conv2D(256, kernel_size=kernel_size, activation='relu'))    
    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(35, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model


def model_5():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu'))    
    model.add(Conv2D(128, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(35, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model