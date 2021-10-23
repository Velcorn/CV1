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


# Create a contrast map of an integral image
def create_contrast_map(image, cs, ss):
    # Create black image according to image shape
    y, x = image.shape
    contrast_map = np.zeros((y, x))

    # Get distance from center pixel to border of windows
    cd, sd = cs // 2, ss // 2

    # Iterate over pixels in bound of surround distance, compute center- and surround average,
    # and set surround average - center average in contrast_map
    for i in range(sd, y-sd):
        for j in range(sd, x-sd):
            ca = integrate(image, (i-cd, j-cd), (i+cd, j+cd)) / cs ** 2
            sa = integrate(image, (i-sd, j-sd), (i+sd, j+sd)) / ss ** 2
            contrast_map[i, j] = sa - ca

    # Return contrast map without borders
    return contrast_map[sd:y-sd, sd:x-sd]


if __name__ == "__main__":
    # Read image, convert to gray and compute integral image
    img = imread("visual_attention_ds.png")
    img_gray = rgb2gray(rgba2rgb(img))
    img_integral = integral_image(img_gray)

    # Create and display contrast map for each center+surround size.
    center_surround_sizes = [[11, 21], [3, 7], [31, 51]]
    for css in center_surround_sizes:
        center_size, surround_size = css[0], css[1]
        img_contrast = create_contrast_map(img_integral, center_size, surround_size)
        plt.imshow(img_contrast, cmap="gray")
        plt.show()

    '''
    How does the resulting contrast map change (upon changing window sizes)? 
    Window size should roughly match the size of the object to highlight, i.e., 
    smaller size highlights smaller objects, larger size highlights larger objects.
    '''
