"""
Chams Alassil Khoury, Adrian Westphal and Jan Willruth
"""

import numpy as np


# Unpickle function from CIFAR-10 dataset website (https://www.cs.toronto.edu/~kriz/cifar.html)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Extract specified number of images/data based on given label
def extract_images(number, data, labels, label):
    # Get specified number of images that match given label
    images = data[labels == label][:number]

    # Reshape into RGB format and transform to float for calculations
    return images.reshape((number, 3, -1)).astype(float)


def calc_distance(a, b):
    return np.linalg.norm(a - b)


if __name__ == "__main__":
    # Extract data_batch_1/train_batch and the contained labels and data into variables
    data_batch_1 = unpickle("CIFAR-10/data_batch_1")
    train_data = np.asarray(data_batch_1[b"data"])
    train_labels = np.asarray(data_batch_1[b"labels"])
    test_batch = unpickle("CIFAR-10/test_batch")
    test_data = np.asarray(test_batch[b"data"])
    test_labels = np.asarray(test_batch[b"labels"])

    # Extract train/test images for labels 1, 4 and 8
    auto_train = extract_images(30, train_data, train_labels, 1)
    deer_train = extract_images(30, train_data, train_labels, 4)
    ship_train = extract_images(30, train_data, train_labels, 8)
    auto_test = extract_images(10, test_data, test_labels, 1)
    deer_test = extract_images(10, test_data, test_labels, 4)
    ship_test = extract_images(10, test_data, test_labels, 8)

    # Get train/test grayscale images
    auto_train_gray = np.sum(auto_train, axis=1) / 3
    deer_train_gray = np.sum(deer_train, axis=1) / 3
    ship_train_gray = np.sum(ship_train, axis=1) / 3
    auto_test_gray = np.sum(auto_test, axis=1) / 3
    deer_test_gray = np.sum(deer_test, axis=1) / 3
    ship_test_gray = np.sum(ship_test, axis=1) / 3

    # Calculate histograms from train/test grayscale images
    bins = [51, 2, 10, 255]
    for bins in bins:
        auto_train_hists = [np.histogram(image, bins, [0, 255])[0] for image in auto_train_gray]
        deer_train_hists = [np.histogram(image, bins, [0, 255])[0] for image in deer_train_gray]
        ship_train_hists = [np.histogram(image, bins, [0, 255])[0] for image in ship_train_gray]
        auto_test_hists = [np.histogram(image, bins, [0, 255])[0] for image in auto_test_gray]
        deer_test_hists = [np.histogram(image, bins, [0, 255])[0] for image in deer_test_gray]
        ship_test_hists = [np.histogram(image, bins, [0, 255])[0] for image in ship_test_gray]

        # Merge train hists and get L_2 distance for every test image
        test_hists = auto_test_hists+deer_test_hists+ship_test_hists

