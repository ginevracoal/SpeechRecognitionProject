from preprocess import *
from CNN_models import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import model_from_json


DATA_PATH = "/galileo/home/userexternal/gcarbone/group/data/speech/"
out_dir = "/galileo/home/userexternal/gcarbone/group/keras_CNN/arrays/"

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

model_1 = model_1()
model_1.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
save_model(model_1,'model_1')

model_2 = model_1()
model_2.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
save_model(model_2,'model_2')



