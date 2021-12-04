"""
Group 01
Chams Alassil Khoury, 7161852
Adrian Westphal, 6940017
Jan Willruth, 6768273
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

# Read image and convert to grayscale
img = imread("coins.jpg")
img_gray = rgb2gray(img)

# Calculate radius in pixels from diameter table for each coin
res = .12  # mm/pixel
diameter_mm = {"10 penni": 16.3, "50 penni": 19.7, "1 mark": 22.25, "5 marks": 24.5, "10 marks": 27.25}
radius_pxl = {k: v / res / 2 for k, v in diameter_mm.items()}

# Apply canny edge detector and visualize edges
edges = canny(img_gray)
plt.imshow(edges, cmap="gray")
plt.show()
plt.close()

"""
Coins detected? Check!
"""

# Calculate Hough transform of the edge detection results and plot result for individual radii
coins, radii = list(radius_pxl.keys()), list(radius_pxl.values())
hough_circles = hough_circle(edges, radii)
for i, hc in enumerate(hough_circles):
    plt.title(f"{coins[i]}")
    plt.imshow(hc, cmap="gray")
    plt.show()
    plt.close()

# Extract peaks from Hough transform
accums, cx, cy, radii = hough_circle_peaks(hough_circles, radii, num_peaks=2, total_num_peaks=10, normalize=True)

# Superimpose circles on the original image
# Drawing of circles "inspired" by
# https://stackoverflow.com/questions/34902477/drawing-circles-on-image-with-matplotlib-and-numpy
fig, ax = plt.subplots(1)
ax.imshow(img)
for cx, cy, r in zip(cx, cy, radii):
    circle = Circle((cx, cy), r, alpha=.2, color="red")
    ax.add_patch(circle)
plt.show()
plt.close()
