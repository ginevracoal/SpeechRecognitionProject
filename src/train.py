import os
import sys
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

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)
    
    @property
    def sampleshape(self):
        return self.x[0].shape


def load_data(dirname, func_name, n_classes):
    print('Loading data from the filesystem ({})'.format(dirname))

    if func_name == 'spectrogram':
        transform = preprocessing.wav2spectrogram
    elif func_name == 'lgspectrogram':
        transform = preprocessing.wav2lgspectrogram
    elif func_name == 'mfcc':
        transform = preprocessing.wav2mfcc
    elif func_name == 'mfcc_tl':
        transform = preprocessing.wav2mfcc_tl
    else:
        raise NotImplementedError

    dataset = {'x': [], 'y': []}
    for root, dirs, files in os.walk(dirname):
        currentdir = os.path.basename(root)
        if not currentdir.startswith('_') and root != dirname:

            cachefile = os.path.join(root, 'vector_' + func_name + '.npy')
            if os.path.exists(cachefile):
                vector = np.load(cachefile)
                print("Read", cachefile)
            else:
                vector = []
                for filename in files:
                    if filename.endswith('.wav'):
                        filepath = os.path.join(root, filename)
                        vector.append(transform(filepath))

                np.save(cachefile, vector)
                print("Written", cachefile)

            outclass = keras.utils.to_categorical(utils.class2int_map[currentdir], n_classes)

            dataset['x'].extend(vector)
            dataset['y'].extend([outclass for i in range(len(vector))])

    print('Loaded {} samples'.format(len(dataset['x'])))

    return dataset


def build_model(input_shape, n_output, name='simple'):
    if name == 'simple':
        return models.build_simple(input_shape, n_output)
    elif name == 'tfclone':
        return models.build_tfclone(input_shape, n_output)
    elif name == 'trlrn_resnet':
        return models.build_trlrn_resnet(input_shape, n_output)
    elif name == 'trlrn_mobilenet':
        return models.build_trlrn_mobilenet(input_shape, n_output)
    elif name == 'trlrn_vgg':
        return models.build_trlrn_vgg(input_shape, n_output)
    else:
        raise NotImplementedError


config = {
    'data_path': '/galileo/home/userexternal/ffranchi/speech',
    'n_classes': 35,
    'split_seed': 44,
    'data_func': 'mfcc',
    'model_name': 'simple',
    'epochs': 50
}

if len(sys.argv) == 3:
    config['data_func'] = sys.argv[1]
    config['model_name'] = sys.argv[2]

dataset = load_data(config['data_path'], config['data_func'], config['n_classes'])

train_x, test_x, train_y, test_y = \
        train_test_split(dataset['x'], dataset['y'], test_size=.2, random_state=config['split_seed'])

train_set = SamplesVector(train_x, train_y, config['data_func'])
test_set = SamplesVector(test_x, test_y, config['data_func'])

model = build_model(train_set.sampleshape, config['n_classes'], name=config['model_name'])
model.summary()

model.fit_generator(train_set, validation_data=test_set, epochs=config['epochs'])

utils.save_model(model, '_'.join((config['model_name'], config['data_func'])))

