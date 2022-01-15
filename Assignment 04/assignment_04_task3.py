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
img_segmented = slic(img, n_segments=len(gt_segments), compactness=10, start_label=1)
img_segments = np.unique(img_segmented)
plt.title("SLIC segmentation")
# plt.show()
plt.close()

# Calculate undersegmentation error:
total_error = 0

# Get segment coordinates in segmented image
seg_coords = {}
for seg in img_segments:
    seg_coords[seg] = list(zip(*np.where(img_segmented == seg)))

# Iterate over segments
for gts in gt_segments:
    # Get coordinates and area for ground truth label
    gt_coords = set(zip(*np.where(label == gts)))
    gt_area = len(gt_coords)

    # Iterate over image segments
    seg_area = 0
    for seg in img_segments:
        coords = seg_coords[seg]
        # If any overlap exists, add to segmentation area
        if any(coord in gt_coords for coord in coords):
            seg_area += len(coords)

    # Calculate error for ground truth segment and increment total error
    error = (seg_area - gt_area) / gt_area
    total_error += error

    # Print segment error
    print(f"Error for segment {gts:02n}: {round(error, 2)}")

# Calculate and print average error over ground truth segments
print(f"Average undersegmentation error: {round(total_error / len(gt_segments), 2)}")


"""
Q: How does the average undersegmentation error change when you increase the desired
number of superpixels n? Why?
A: It increases because more wrongly labeled superpixels overlap with the ground truth pixels.
"""
