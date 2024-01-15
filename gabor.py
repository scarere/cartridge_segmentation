from skimage.filters import gabor_kernel
from skimage.transform import rescale, resize
import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import ndimage as ndi
import cv2

mm9 = 'Cwrbf0001-9mm.png'
mm4010 = 'Lightstone-40-10mm.png'

img = imread(mm4010)

# crop image to a square and downsample
mindim = np.argmin(img.shape)
maxdim = np.argmax(img.shape)
offset = int(abs(img.shape[0] - img.shape[1])/2)
if mindim == 1:
  img = img[offset:offset+img.shape[mindim], :img.shape[mindim]]
else:
  img = img[:img.shape[mindim], offset:offset+img.shape[mindim]]

img = resize(img, (128, 128))
img = img*255

# kernel = np.real(gabor_kernel(theta=0, frequency=0.2, sigma_x=1, sigma_y=1))
# filtered = ndi.convolve(img, kernel, mode='wrap')
# filtered = cv2.blur(filtered, (3, 3))
mask = filtered.copy()
mean = np.mean(mask)
t = mean*1.5
mask[mask < t] = 0
mask[mask > t] = 1
plt.imshow(filtered, cmap='gray')
plt.show()