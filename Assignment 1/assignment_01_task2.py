"""
Chams Alassil Khoury, 7161852
Adrian Westphal,
Jan Willruth, 6768273
"""

from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform.integral import integral_image, integrate
import matplotlib.pyplot as plt
import numpy as np


# Create a contrast map an integral image
def create_contrast_map(image, center, surround):
    # Create black image according to image size
    contrast_map = np.zeros(image.shape)

    # Iterate over pixels in bound of surround size, compute c and s, and add to contrast_map
    for x in range(surround, image.shape[0] - surround):
        for y in range(surround, image.shape[1] - surround):
            c = integrate(image, (x, y), (x + center, y + center)) / center * center
            s = integrate(image, (x, y), (x + surround, y + surround)) / surround * surround
            contrast_map[x, y] = s-c

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
    How does the resulting contrast map change?
    Obviously gets smaller, but also loses details (gets more blurry).
    '''
