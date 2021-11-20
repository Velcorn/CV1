"""
Chams Alassil Khoury, 7161852
Adrian Westphal, 6940017
Jan Willruth, 6768273
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor

# Load data
with open("noisyedgepoints.npy" "rb") as f:
    X = np.load(f)
    y = np.load(f)

# Use RANSACRegressor to find the best fitting line and plot everything
reg = RANSACRegressor()

