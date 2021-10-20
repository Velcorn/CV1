"""
Chams Alassil Khoury, Adrian Westphal and Jan Willruth
"""

import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def extract_images(amount, data, labels, label):
    return np.asarray(data)[np.asarray(labels) == label][:amount]


def euclidian_distance(a, b):
    return np.linalg.norm(a - b)


if __name__ == "__main__":
    data_batch_1 = unpickle("CIFAR-10/data_batch_1")
    labels = data_batch_1[b"labels"]
    data = data_batch_1[b"data"]

    automobile = extract_images(30, data, labels, 1)
    print(automobile[0])
