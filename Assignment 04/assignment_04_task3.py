"""
Group 01
Chams Alassil Khoury, 7161852
Adrian Westphal, 6940017
Jan Willruth, 6768273
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.segmentation import slic

# Load images
img = imread("0001_rgb.png")
label = imread("0001_label.png")

# Apply SLIC to input and visualize
# start_label = 1 to "suppress" FutureWarning
gt_segments = np.unique(label[label > 0])
img_segments = [x for x in range(1, np.max(gt_segments) + 1)]
img_seg = slic(img, n_segments=len(img_segments), compactness=10, start_label=1)
plt.imshow(img_seg)
plt.title("SLIC segmentation")
plt.show()
plt.close()

# Calculate undersegmentation error
total_error = 0
# Iterate over segments
for gts in gt_segments:
    # Get coordinates and area for ground truth label
    gt_coords = np.where(label == gts)
    gt_coords = set(zip(*gt_coords))
    gt_area = np.count_nonzero(label == gts)

    # Iterate over image segments
    seg_area = 0
    for imgs in img_segments:
        # Get coordinates for image segment
        imgs_coords = np.where(img_seg == imgs)
        imgs_coords = list(zip(*imgs_coords))
        # If any overlap exists, add to segmentation area
        if any(coord in gt_coords for coord in imgs_coords):
            seg_area += len(imgs_coords)
    # Calculate error for ground truth segment and increment total error
    error = (seg_area - gt_area) / gt_area
    total_error += error

    # Print segment error
    print(f"Error for segment {gts:02n}: {round(error, 2)}")

# Calculate and print average error over ground truth segments
print(f"Average segmentation error: {round(total_error / len(gt_segments), 2)}")


"""
Q: How does the average undersegmentation error change when you increase the desired
number of superpixels n? Why?
A: It gets lower because there are more segments/labels than there are in the ground truth, resulting in less area 
for the sum which in turn results in smaller (more likely to be negative) numerators and thus smaller error(s).
"""
