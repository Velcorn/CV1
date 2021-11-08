"""
Chams Alassil Khoury, 7161852
Adrian Westphal, 6940017
Jan Willruth, 6768273
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform.pyramids import pyramid_gaussian

img = imread("visual_attention.png")
img_gray = rgb2gray(img)

# Pyramid stuff
