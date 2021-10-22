"""
Chams Alassil Khoury, 7161852
Adrian Westphal,
Jan Willruth, 6768273
"""
import sys

from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform.integral import integral_image, integrate
import matplotlib.pyplot as plt
import numpy as np


# Create a contrast map an integral image
def create_contrast_map(image, c, s):
    # Create black image according to image size
    y, x = image.shape
    contrast_map = np.zeros((y, x))

    # Iterate over pixels in bound of surround size, compute c and s, and add s-c to contrast_map
    for i in range(s, y - s):
        for j in range(s, x - s):
            center = integrate(image, (i, j), (i + c, j + c)) / c * c
            surround = integrate(image, (i, j), (i + s, j + s)) / s * s
            contrast_map[i+s, j+s] = surround - center

    return contrast_map


if __name__ == "__main__":
    # Read image, convert to gray and compute integral image
    img = imread("visual_attention_ds.png")
    img_gray = rgb2gray(rgba2rgb(img))
    img_integral = integral_image(img_gray)

    # Create and display contrast map for each center+surround size.
    center_surround = [[11, 21]]  # [[11, 21], [3, 7], [31, 51]]
    for cs in center_surround:
        img_contrast = create_contrast_map(img_integral, cs[0]-1, cs[1]-1)
        plt.imshow(img_contrast, cmap="gray")
        plt.show()

    '''
    How does the resulting contrast map change?
    Obviously gets smaller, but also loses details (gets more blurry).
    '''
