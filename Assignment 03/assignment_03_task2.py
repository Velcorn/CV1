"""
Group 01
Chams Alassil Khoury, 7161852
Adrian Westphal, 6940017
Jan Willruth, 6768273
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor

# Load data
with open("noisyedgepoints.npy", "rb") as f:
    X = np.load(f)
    y = np.load(f)

# Use RANSACRegressor to find the best fitting line
ransac = RANSACRegressor()
ransac.fit(X, y)
inliers = ransac.inlier_mask_
outliers = ~inliers
line_X = np.arange(min(X), max(X))[:, np.newaxis]
line_y = ransac.predict(line_X)

# Plot model
fig, ax = plt.subplots()
size = 1
ax.scatter(X[inliers], y[inliers], s=size, color="black", label="Inliers", marker="o")
ax.scatter(X[outliers], y[outliers], s=size, color="red", label="Outliers", marker="X")
ax.plot(line_X, line_y, color="blue", label="RANSAC edge proposal")
plt.title("RANSAC Edge Model")
plt.legend(loc="lower right")
plt.show()
