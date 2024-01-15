from skimage.transform import resize
from skimage.feature import peak_local_max
import skimage.segmentation as seg
import skimage.filters as filters
import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
import cv2
from functions import to_square, threshold_binarize, seperate_mask_areas, overlay_multiclass_mask, seperate_multiclass_mask, find_drag
from scipy import ndimage as ndi
from matplotlib import patches

def main(image_path):
  # Load image
  img = imread(image_path)

  # crop image to a square
  img = to_square(img)

  # Downsample image
  img = resize(img, (128, 128))

  # Convert pixel intensities from 0-1 to 0-255
  img = img*255

  # make a copy of the original image to start doing some processing on
  mask = img.copy()

  #smooth mask to improve thresholding
  mask = cv2.blur(mask, (5, 5))

  #threshold
  mean = np.mean(mask)
  t = mean*0.9
  mask = threshold_binarize(img=mask, thresh=t)

  # Remove objects/ROI's that are connected to the image border
  mask = seg.clear_border(mask)

  # Seperate ROI's based on connectivity
  rois = seperate_mask_areas(mask, return_sizes=False)

  # Assume breach face is largest roi and firing pin is second largest
  bf = rois[0]
  fp = rois[1]

  # Now to try and seperate the aperture shear from the breach face roi
  # Nothing I have tried has worked.
  # Hough circles, filters of various kinds, thresholding etc. all didn't seperate aperture shear well enough to differentiate from breach face
  # bfimg = np.where(bf, img, 0)
  # o = filters.meijering(bfimg)
  # o = filters.median(o)
  # o = filters.threshold_local(o)
  # plt.imshow(o, cmap='gray')
  # plt.show()

  # Firing Pin Drag
  drag, coords = find_drag(bf)

  # Create final mask with multiple labels
  mask = np.zeros(bf.shape)
  mask = np.where(bf, 1, mask)
  mask = np.where(fp, 2, mask)
  mask = np.where(drag, 3, mask)

  # Plot an overlay of the class segmentations
  labels = ['Breach Face', 'Firing Pin Impression', 'Firing Pin Drag', 'Direction of Firing Pin Drag']
  colors = ['red', 'purple', 'cyan', 'blue']
  overlay_multiclass_mask(image=img, mask=mask, classes=labels, colors=colors, show=False)

  if coords:
    p1 = coords[0]
    p2 = coords[1]
    plt.arrow(p1[1], p1[0], (p2[1] - p1[1])*2, (p2[0] - p1[0])*2, width=1, facecolor='blue', fill=True, edgecolor='blue')

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  # Various different images
  mm9 = 'Cwrbf0001-9mm.png' # 9mm, negligible (very small) firing pin drag
  mm4010 = 'Lightstone-40-10mm.png' # 40/10mm, large firing pin drag present
  main(mm9)
  main(mm4010)
