from sklearn.model_selection import train_test_split


def load_data(dirname):
    pass

def build_model():
    pass

def train(model, x, y, checkpoint_every=100):
    pass

def validate(model, x, y)

dataset = load_data('~/speech')

seed = 44
train_x, test_x, train_y, test_y = \
        train_test_split(dataset['x'], dataset['y'], test_size=.2, random_state=seed)

model = build_model()

train()

results = validate()
