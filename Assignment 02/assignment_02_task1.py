"""
Chams Alassil Khoury, 7161852
Adrian Westphal, 6940017
Jan Willruth, 6768273
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import match_template

template = imread("coco264316clock.jpg")
img = imread("coco264316.jpg")

template_gray = rgb2gray(template)
img_gray = rgb2gray(img)

correction = match_template(img_gray, template_gray)
plt.imshow(correction, cmap="gray")
plt.show()

bp = np.where(correction == np.max(correction))
print(f"Brightest pixel at coordinates: {bp[0][0]}, {bp[1][0]}!")

'''
Looks good!
'''

template_gray_flipped = np.fliplr(template_gray)
correction_flipped = match_template(img_gray, template_gray_flipped)
plt.imshow(correction_flipped, cmap="gray")
plt.show()

bpf = np.where(correction_flipped == np.max(correction_flipped))
print(f"Brightest pixel at coordinates: {bpf[0][0]}, {bpf[1][0]}!")

'''
Looks less good ^^
'''
