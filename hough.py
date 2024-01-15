from skimage.filters import gabor_kernel
from skimage.transform import rescale, resize
import cv2
import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
from matplotlib import patches

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

# smooth image
smoothed = cv2.blur(img, (3, 3))

# detect circles (algorithm requires uint8)
smoothed = smoothed.astype(np.uint8)
circles = cv2.HoughCircles(
  smoothed, 
  method=cv2.HOUGH_GRADIENT, 
  dp=1, 
  minDist=40,
  param1=60,
  param2=40,
  minRadius=20,
  maxRadius=0
  )

fig, ax = plt.subplots()
ax.imshow(smoothed, cmap='gray')

if circles is not None:
  circles = circles[0]
  for circle in circles:
    a, b, r = circle
    print(circle)
    ax.add_patch(patches.Circle((a, b), r, color='red', fill=False, linewidth=2))
    
plt.show()
