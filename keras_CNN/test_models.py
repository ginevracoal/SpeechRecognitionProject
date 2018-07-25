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
import librosa
import pandas as pd


DATA_PATH = "/galileo/home/userexternal/gcarbone/group/data/speech/"
out_dir = "/galileo/home/userexternal/gcarbone/group/keras_CNN/arrays_32/"

labels, _, _ = get_labels()

## Loading train set and test set
X, X_test, y, y_test = get_train_test()

# # taking a subset to test the job
# n = 300
# X = X[:n]
# X_test = X_test[:n//2]
# y = y[:n]
# y_test = y_test[:n//2]

## Feature dimension
feature_dim_1 = 32
feature_dim_2 = 32
channel = 1
epochs = 20
batch_size = 100
verbose = 1
num_classes = len(labels) # 35 without background noise

## Reshaping to perform 2D convolution, which is spatial convolution over images
X = X.reshape(X.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
y_hot = to_categorical(y)
y_test_hot = to_categorical(y_test)

# split train into train and validation
train_len = len(X)//2

X_train = X[:train_len]
y_train_hot = y_hot[:train_len]

X_validation = X[train_len:]
y_validation_hot = y_hot[train_len:]

print(X_train.shape, y_train_hot.shape)
print(X_validation.shape, y_validation_hot.shape)


## with this piece of code I can give the name as an input both from command line and as console input ;)
if __name__ == "__main__":
		try:
				model_name = sys.argv[1]
		except IndexError:
				model_name = input("\nPlease give the name of the model: ")

print("\nYou entered ", model_name)

model = eval(model_name)()
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_validation, y_validation_hot))
save_model(model, model_name)

############ make real predictions

score = model.evaluate(X_test, y_test_hot)
print(score)


# df = pd.read_csv(DATA_PATH+'testing_list.txt', delimiter=',')
# testing_list = df.iloc[:,0]
# testing_list = [os.path.join(DATA_PATH,filename) for filename in testing_list]

# # Predicts one sample given its path
# def predict(filepath, model):
#     sample = wav2mfcc(filepath)
#     sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
#     return get_labels()[0][np.argmax(model.predict(sample_reshaped))]

# def predict_list(testing_list, model):
#     predictions = []
#     for filename in testing_list:
#         predictions.append(predict(filename, model=model))
#     return(predictions)

# predictions = predict_list(testing_list, model)

# ## save the results as a dictionary
# json.dump(predictions, open(path + model_name + '_predictions.json','w'))