import os
import keras
import numpy as np
from sklearn.model_selection import train_test_split

import utils
import models
import preprocessing

class SamplesVector(keras.utils.Sequence):

    def __init__(self, x, y, transformation_type, batch_size=25):
        self.x, self.y = x, y
        self.batch_size = batch_size
        if transformation_type == 'spectrogram':
            self.sampleshape = (129, 71, 1)
            self.transformation_func = preprocessing.wav2spectrogram
        elif transformation_type == 'lgspectrogram':
            self.sampleshape = (1025, 71, 1)
            self.transformation_func = preprocessing.wav2lgspectrogram
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


def build_model(input_shape, n_output, name='simple'):
    if name == 'simple':
        return models.build_simple(input_shape, n_output)
    elif name == 'tfclone':
        return models.build_tfclone(input_shape, n_output)
    elif name == 'trlrn':
        return models.build_trlrn(input_shape, n_output)
    else:
        raise NotImplementedError


config = {
    'data_path': '/galileo/home/userexternal/ffranchi/speech',
    'n_classes': 36,
    'split_seed': 44,
    'data_func': 'spectrogram',
    'model_name': 'simple'
}

dataset = load_data(config['data_path'])

train_x, test_x, train_y, test_y = \
        train_test_split(dataset['x'], dataset['y'], test_size=.2, random_state=config['split_seed'])

train_set = SamplesVector(train_x, train_y, config['data_func'])
test_set = SamplesVector(test_x, test_y, config['data_func'])

model = build_model(train_set.sampleshape, config['n_classes'], name=config['model_name'])
model.summary()

model.fit_generator(train_set, validation_data=test_set)

utils.save_model(model, '_'.join((config['model_name'], config['data_func'])))

