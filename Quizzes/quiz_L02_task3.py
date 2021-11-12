import numpy as np
from skimage.transform.integral import integral_image

img = np.asarray([[5, 1, 4, 1, 3],
                  [5, 2, 3, 7, 2],
                  [2, 3, 1, 2, 3],
                  [3, 4, 1, 2, 3],
                  [2, 4, 5, 6, 2]])

print(integral_image(img))
