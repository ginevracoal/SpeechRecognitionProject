import os
from sklearn.model_selection import train_test_split

import utils

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


def build_model():
    pass

def train(model, x, y, checkpoint_every=100):
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
