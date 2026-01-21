import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

