
def save_model(model, filename):
    print("Saving model and weight")
    with open(filename + '.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(filename + '.h5')

class2int_map = {
    'speech': 0,
    'bed': 1,
    'go': 2,
    'zero': 3,
    'bird': 4,
    'yes': 5,
    'four': 6,
    'eight': 7,
    'left': 8,
    'nine': 9,
    'up': 10,
    'no': 11,
    'six': 12,
    'two': 13,
    'right': 14,
    'tree': 15,
    'stop': 16,
    'cat': 17,
    'learn': 18,
    'dog': 19,
    'marvin': 20,
    'on': 21,
    'seven': 22,
    'backward': 23,
    'down': 24,
    'happy': 25,
    'one': 26,
    'sheila': 27,
    'visual': 28,
    'follow': 29,
    'wow': 30,
    'off': 31,
    'three': 32,
    'house': 33,
    'five': 34,
    'forward': 35
}

class2int_inv_map = {v: k for k, v in class2int_map.items()}
