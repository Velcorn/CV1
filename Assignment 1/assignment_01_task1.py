"""
Chams Alassil Khoury,
Adrian Westphal
Jan Willruth, 6768273
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


# Convert an array of image data to grayscale
def to_grayscale(array):
    return np.sum(array, axis=1) / 3


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
    auto_label, deer_label, ship_label = 1, 4, 8
    auto_train = extract_images(30, train_data, train_labels, auto_label)
    deer_train = extract_images(30, train_data, train_labels, deer_label)
    ship_train = extract_images(30, train_data, train_labels, ship_label)
    auto_test = extract_images(10, test_data, test_labels, auto_label)
    deer_test = extract_images(10, test_data, test_labels, deer_label)
    ship_test = extract_images(10, test_data, test_labels, ship_label)

    # Get train/test grayscale images
    auto_train_gray = to_grayscale(auto_train)
    deer_train_gray = to_grayscale(deer_train)
    ship_train_gray = to_grayscale(ship_train)
    auto_test_gray = to_grayscale(auto_test)
    deer_test_gray = to_grayscale(deer_test)
    ship_test_gray = to_grayscale(ship_test)

    # Calculate histograms from train/test grayscale images; attach label for later
    bin_sizes = [2, 10, 51, 255]
    for bin_size in bin_sizes:
        auto_train_hists = [[auto_label, np.histogram(image, bin_size, [0, 255])[0]] for image in auto_train_gray]
        deer_train_hists = [[deer_label, np.histogram(image, bin_size, [0, 255])[0]] for image in deer_train_gray]
        ship_train_hists = [[ship_label, np.histogram(image, bin_size, [0, 255])[0]] for image in ship_train_gray]
        auto_test_hists = [[auto_label, np.histogram(image, bin_size, [0, 255])[0]] for image in auto_test_gray]
        deer_test_hists = [[deer_label, np.histogram(image, bin_size, [0, 255])[0]] for image in deer_test_gray]
        ship_test_hists = [[ship_label, np.histogram(image, bin_size, [0, 255])[0]] for image in ship_test_gray]

        # Merge train/test hists
        train_hists = auto_train_hists+deer_train_hists+ship_train_hists
        test_hists = auto_test_hists+deer_test_hists+ship_test_hists

        # Calculate accuracy by iterating over the test_hists, calculating all distances with the train_hists and
        # comparing the label of the train_hist with the lowest distance to that of the test_hist
        total = len(test_hists)
        correct = 0
        for test_hist in test_hists:
            # Calc distances and get index of min distance
            distances = np.asarray([calc_distance(test_hist[1], train_hist[1]) for train_hist in train_hists])
            min_index = np.where(distances == np.min(distances))[0][0]

            # Zip labels to distances for comparison
            reference = list(zip([train_hist[0] for train_hist in train_hists], distances))
            if test_hist[0] == train_hists[min_index][0]:
                correct += 1

        accuracy = round(100*correct/total, 2)
        print(f"Classification accuracy (bins = {bin_size}): {(len(str(max(bin_sizes)))-len(str(bin_size)))*' '}"
              f"{accuracy}%")
