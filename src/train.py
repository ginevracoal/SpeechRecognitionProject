import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

import utils

sample_shape = (1025, 71, 1)

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

    print('Converting classes to integers')
    dataset['y'] = [utils.class2int_map[x] for x in dataset['y']]

    return dataset


def build_simple_model():
    n_out_class = len(utils.class2int_map)
    model = Sequential()

    model.add(Conv2D(64, (20, 8), activation='relu', input_shape=sample_shape))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (10, 4), activation='relu', input_shape=sample_shape))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dense(n_out_class, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model


def build_trlrn_model():
    raise NotImplementedError


def build_model(name='simple'):
    if name == 'simple':
        return build_simple_model()
    elif name == 'trlrn':
        return build_trlrn_model()
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

model = build_model()

train(model, train_x, train_y)

results = validate(model, test_x, test_y)
