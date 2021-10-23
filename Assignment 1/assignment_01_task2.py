"""
Chams Alassil Khoury, 7161852
Adrian Westphal,
Jan Willruth, 6768273
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgba2rgb, rgb2gray
from skimage.io import imread
from skimage.transform.integral import integral_image, integrate


# Create a contrast map an integral image
def create_contrast_map(image, c, s):
    # Create black image according to image shape without surround box borders
    y, x = image.shape
    contrast_map = np.zeros((y-s, x-s))

    # Iterate over pixels in bound of surround size, compute c and s, and set s-c in contrast_map
    for i in range(s, y):
        for j in range(s, x):
            center = integrate(image, (i-c, j-c), (i, j)) / c**2
            surround = integrate(image, (i-s, j-s), (i, j)) / s**2
            contrast_map[i-s, j-s] = surround - center

    return contrast_map


if __name__ == "__main__":
    # Read image, convert to gray and compute integral image
    img = imread("visual_attention_ds.png")
    img_gray = rgb2gray(rgba2rgb(img))
    img_integral = integral_image(img_gray)

    # Create and display contrast map for each center+surround size.
    center_surround = [[11, 21], [3, 7], [31, 51]]
    for cs in center_surround:
        img_contrast = create_contrast_map(img_integral, cs[0], cs[1])
        plt.imshow(img_contrast, cmap="gray")
        plt.show()

    '''
    How does the resulting contrast map change (upon changing window sizes)? 
    Window size should roughly match the size of the object to highlight, i.e., 
    smaller size highlights smaller objects, larger size highlights larger objects.
    '''
