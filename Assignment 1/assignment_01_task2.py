from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import integral_image, integrate
import matplotlib.pyplot as plt

img = imread("visual_attention_ds.png")
img_gray = rgb2gray(rgba2rgb(img))
integral = integral_image(img)

plt.imshow(img_gray, cmap="gray")
plt.show()
