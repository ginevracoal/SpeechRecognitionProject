from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

labels, _, _ = get_labels()

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
feature_dim_1 = 20
feature_dim_2 = 11
channel = 1
epochs = 50
batch_size = 100
verbose = 1
num_classes = len(labels) # 35 without background noise

# Reshaping to perform 2D convolution, which is spatial convolution over images
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

def get_model():
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

# Predicts one sample given its filepath from current folder
def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]

# predicts a set of samples (given the matrices themselves)
def predict_samples(test_samples, model):
    predictions = []
    for sample in test_samples:
        sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
        predictions.append(get_labels()[0][
                np.argmax(model.predict(sample_reshaped))])
    return predictions

# predicts given the list of filepaths from the current folder
def predict_list(testing_list, model):
    predictions = []
    for filename in testing_list:
        predictions.append(predict(filename, model=model))
    return(predictions)


