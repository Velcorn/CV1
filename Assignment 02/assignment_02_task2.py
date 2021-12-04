"""
Group 01
Chams Alassil Khoury, 7161852
Adrian Westphal, 6940017
Jan Willruth, 6768273
"""

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import random_noise
from skimage.filters import gaussian, sobel

img = imread("woman.png")

# Apply gaussian (noise and smoothing)
N = random_noise(img, mode="gaussian", var=0.01)
S = gaussian(N, sigma=1.0)

# Sobel filter and visualization
F_n = sobel(N)
plt.title("F_n")
plt.imshow(F_n, cmap="gray")
plt.show()
F_s = sobel(S)
plt.title("F_s")
plt.imshow(F_s, cmap="gray")
plt.show()

# Intensity hists
plt.hist(F_n.ravel(), bins="auto")
plt.title("Hist F_n")
plt.show()
plt.hist(F_s.ravel(), bins="auto")
plt.title("Hist F_s")
plt.show()

# Picked threshold values from the plotted hists
t_n, t_s = .12, .06

# Apply and visualize binary masks
mask_F_n = F_n
mask_F_n[mask_F_n < t_n] = 0
mask_F_n[mask_F_n > t_n] = 1
plt.title("Mask F_n")
plt.imshow(mask_F_n, cmap="gray")
plt.show()

mask_F_s = F_s
mask_F_s[mask_F_s < t_s] = 0
mask_F_s[mask_F_s > t_s] = 1
plt.title("Mask F_s")
plt.imshow(mask_F_s, cmap="gray")
plt.show()

'''
Did you notice it is almost impossible to tune the threshold for the noisy image to show
just the outline of the face?
Yes ^^'

Why is edge detection improved by applying smoothing?
Noise affects derivatives and thus the effectivity of the filter(s).
'''
