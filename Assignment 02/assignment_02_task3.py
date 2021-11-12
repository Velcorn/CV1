"""
Chams Alassil Khoury, 7161852
Adrian Westphal, 6940017
Jan Willruth, 6768273
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import pyramid_gaussian
from skimage.transform import resize

img = imread("visual_attention.png")
img_gray = rgb2gray(rgba2rgb(img))

# Center and surround pyramids and visualization
max_layer = 4
sigma_c, sigma_s = 9, 16
pyramid_center = list(pyramid_gaussian(img_gray, max_layer=max_layer, sigma=sigma_c))
pyramid_surround = list(pyramid_gaussian(img_gray, max_layer=max_layer, sigma=sigma_s))

for C_j in pyramid_center:
    plt.imshow(C_j, cmap="gray")
    plt.show()

for S_j in pyramid_surround:
    plt.imshow(S_j, cmap="gray")
    plt.show()

# on-off and off-on pyramids calculation and visualization
pyramid_on_off = [np.clip(pyramid_surround[i]-pyramid_center[i], 0, 1) for i in range(max_layer)]
pyramid_off_on = [np.clip(pyramid_center[i]-pyramid_surround[i], 0, 1) for i in range(max_layer)]

for l_on_off in pyramid_on_off:
    plt.imshow(l_on_off, cmap="gray")
    plt.show()

for l_off_on in pyramid_off_on:
    plt.imshow(l_off_on, cmap="gray")
    plt.show()

# Upsample layers, calculate feature map and visualize
for i in range(1, max_layer):
    pyramid_on_off[i] = resize(pyramid_on_off[i], pyramid_on_off[0].shape)
    pyramid_off_on[i] = resize(pyramid_off_on[i], pyramid_off_on[0].shape)

feature_map_on_off = np.mean(pyramid_on_off, axis=0)
feature_map_off_on = np.mean(pyramid_off_on, axis=0)

plt.imshow(feature_map_on_off, cmap="gray")
plt.show()
plt.imshow(feature_map_off_on, cmap="gray")
plt.show()

# Saliency map calculation and visualization
saliency_map = np.mean([feature_map_on_off, feature_map_off_on], axis=0)

plt.imshow(saliency_map, cmap="gray")
plt.show()

'''
What advantage does using this approach based on image pyramids have over the integral
image method from assignment sheet 1?
No need for window sizes that fit the size of the object we want to detect. Also, execution is faster.
'''
