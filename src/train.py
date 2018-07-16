import os
import keras
import numpy as np
from scipy import signal
from scipy.io import wavfile
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split

import utils

class SamplesVector(keras.utils.Sequence):

    def __init__(self, x, y, transformation_type, batch_size=25):
        self.x, self.y = x, y
        self.batch_size = batch_size
        if transformation_type == 'spectrogram':
            self.sampleshape = (1025, 71, 1)
            self.transformation_func = wav2spectrogram
        elif transformation_type == 'mfcc':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        for x in self.x[idx * self.batch_size:(idx + 1) * self.batch_size]:
            sample = self.transformation_func(x)
            padding_size = self.sampleshape[1] - sample.shape[1]
            if padding_size:
                sample = np.pad(sample, ((0, 0), (0, padding_size)), 'constant')
            batch_x.append(sample.reshape(self.sampleshape))

        return np.array(batch_x), np.array(batch_y)
        

def wav2spectrogram(filename):
    sampling_rate, samples = wavfile.read(filename)
    f, t, spectrogram = signal.spectrogram(samples, sampling_rate, nfft=2048)
    return spectrogram

def load_data(dirname):
    print('Loading data from the filesystem ({})'.format(dirname))

    dataset = {'x': [], 'y': []}
    for root, dirs, files in os.walk(dirname):
        currentdir = os.path.basename(root)
        if not currentdir.startswith('_'):
            for filename in files:
                if filename.endswith('.wav'):
                    dataset['x'].append(os.path.join(root, filename))
                    dataset['y'].append(currentdir)

    print('Loaded {} samples'.format(len(dataset['x'])))

    classes = set(dataset['y'])
    print(len(classes), 'classes found: ', ', '.join(classes))

    dataset['y'] = [utils.class2int_map[x] for x in dataset['y']]
    dataset['y'] = keras.utils.to_categorical(dataset['y'])

    return dataset


def build_simple_model(input_shape, n_output):
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


def build_trlrn_model(input_shape, n_output):
    raise NotImplementedError


def build_model(input_shape, n_output, name='simple'):
    if name == 'simple':
        return build_simple_model(input_shape, n_output)
    elif name == 'trlrn':
        return build_trlrn_model(input_shape, n_output)
    else:
        raise NotImplementedError


def train(model, x, y):
    pass

def validate(model, x, y):
    pass

dataset = load_data('/galileo/home/userexternal/ffranchi/speech')

seed = 44
train_x, test_x, train_y, test_y = \
        train_test_split(dataset['x'], dataset['y'], test_size=.2, random_state=seed)

train_set = SamplesVector(train_x, train_y, 'spectrogram')

model = build_model(train_set.sampleshape, 36)
model.summary()

model.fit_generator(train_set)

results = validate(model, test_x, test_y)
