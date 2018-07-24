## EXAMPLE: go to the main folder and run
## python group/keras_CNN/test_models.py model_1 > group/keras_CNN/trained_models/model_1.out

from preprocess import *
from CNN_models import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import model_from_json
import sys

DATA_PATH = "/galileo/home/userexternal/gcarbone/group/data/speech/"
out_dir = "/galileo/home/userexternal/gcarbone/group/keras_CNN/arrays/"

labels, _, _ = get_labels()

## Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

## Feature dimension
feature_dim_1 = 32
feature_dim_2 = 32
channel = 1
epochs = 10
batch_size = 100
verbose = 1
num_classes = len(labels) # 35 without background noise

## Reshaping to perform 2D convolution, which is spatial convolution over images
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

## taking a subset to test the job
# n = 300
# X_train = X_train[:n]
# X_test = X_test[:n//2]
# y_train_hot = y_train_hot[:n]
# y_test_hot = y_test_hot[:n//2]

## with this piece of code I can give the name as an input both from command line and as console input ;)
if __name__ == "__main__":
		try:
				model_name = sys.argv[1]
		except IndexError:
				model_name = input("\nPlease give the name of the model: ")

print("\nYou entered ", model_name)

model = eval(model_name)()
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
save_model(model, model_name)