import numpy as np
from skimage.segmentation import flood_fill
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.color import label2rgb
import skimage.segmentation as seg
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def to_square(img):
  '''Crops an image along it's longest dimension such that it has an equal number of pixels in the x and y dimensions. Crops an equal amount of pixels on each end of the dimension so that the cropped image is centered on the original image

  Args:
    img: A numpy array representing the image where the first two dimensions are the spatial dimensions

  Returns:
    cropped: A cropped version of the input image
  '''

  # crop image to a square
  mindim = np.argmin(img.shape)
  maxdim = np.argmax(img.shape)
  offset = int(abs(img.shape[0] - img.shape[1])/2)
  if mindim == 1:
    cropped = img[offset:offset+img.shape[mindim], :img.shape[mindim]]
  else:
    cropped = img[:img.shape[mindim], offset:offset+img.shape[mindim]]

  return cropped

def threshold_binarize(img, thresh):
  '''Converts an image into a binary mask using a defined threshold. Pixels above the threshold are set to 1, other pixels are set to 0.

  Args:
    img: A numpy array representing the image where the first two dimensions are the spatial dimensions
    thresh: The threshold value for the pixel intensities

  Returns:
    mask: A 2D numpy array representing a the mask image
  '''
  mask = img.copy()
  mask[img > thresh] = 1
  mask[img < thresh] = 0

  return mask

def seperate_multiclass_mask(mask, sort_areas=True):
  '''Given a 2D mask with multiple classes labelled 1 to n_classes (0 is reserved for the background), seperates each class into it's own seperate 2D mask

  Args:
    input_mask: A 2D numpy array representing a multiclass mask where background pixels have a value of 0 and other pixels are labelled using integers from 1 to n_classes
    sort_areas: Whether or not to sort the masks by the number of pixels in each class. Otherwise they are returned in order fron class 1 to class n_classes

  Returns:
    output_masks: A list of 2D numpy arrays representing the seperate mask images for each class
    mask_sizes: A list containing the number of pixels (pixels that are set to True) in each seperate area
  '''

  areas = []
  unique, counts = np.unique(mask, return_counts=True)
  for i, label in enumerate(unique): # iterate through different class labels
    if label != 0: # ignore 0 since this label is reserved for the background
      areas.append([np.where(mask == label, 1, 0), counts[i]]) # append the sub-mask and its size

  # sort masks based on the number of non-zero pixels
  if sort_areas:
    areas.sort(key=lambda x: x[1], reverse=True)

  return [area[0] for area in areas], [area[1] for area in areas]

def seperate_mask_areas(mask, sort_areas=True, return_sizes=True):
  '''Given a binary mask image (either boolean or has values 0 and 1 only), seperates unconnected volumes into seperate mask images

  Args:
    mask: A 2D numpy array representing the mask image. Will be converted into an integer array of 1's and zeros
    sort_areas (optional): Whether to sort the output masks from largest to smallest area size
    return_sizes (optional): Whether to return the pixel counts of each of the seperate areas

  Returns:
    out_masks: A list of 2D numpy arrays representing the seperate mask images for each area
    mask_sizes (optional): A list containing the number of pixels (pixels that are set to True) in each seperate area
  '''

  mask = np.array(mask, dtype=int)

  # Seperate ROI's by giving them different labels based on connectivity
  current_label = 1
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      if mask[i, j] == 1: # Looking for a shape in the mask that hasn't already been relabelled with flood fill
        current_label += 1
        mask = flood_fill(mask, (i, j), current_label) # all connected pixels set to new label

  mask = np.where(mask == 0, 0, mask - 1)

  # Seperate the different classes into seperate masks
  areas, area_sizes = seperate_multiclass_mask(mask, sort_areas=True)
  if return_sizes:
    return areas, area_sizes
  else:
    return areas
  
def overlay_multiclass_mask(image, mask, classes, colors, show=True):
  '''Given a mask image with multiple classes, diisplays the mask overlayed on the input image with each class as a different color. Also creates a legend.

  Args:
    image: The underlay image, assumes grayscale image for now
    mask: A 2D numpy array representing the mask image where 0's represent the background and each class has integer labels starting from 1 to n_classes
    classes: A list of string names for each class in ascending order based on the integer class labels
    colors: A list of matplotlib compatible colors in ascending order based on integer class labels
    show: Whether or not to show the figure or just create it
  '''

  plt.figure()
  plt.imshow(image, cmap='gray') # plot underlay image

  # create patches for legend
  p = []

  for i in range(len(classes)):
    p.append(patches.Patch(color=colors[i], label=classes[i]))

  overlay = label2rgb(label=mask, bg_label=0, colors=colors) # convert label mask into an rgb overlay with different colors for each class
  overlay = np.ma.masked_where(overlay < 1, overlay) # remove background pixels

  plt.imshow(overlay, alpha=0.4)
  plt.legend(handles=p, loc='upper right', bbox_to_anchor=(1.6, 1.0))

  if show:
    plt.tight_layout()
    plt.show()

def find_drag(bf_mask):
  '''Given a mask for the breach face, will isolate the inner area and check to see if a firing pin drag is present

  Args:
    bf_mask: 2D numpy array representing a mask for the breach face area. Should be an area with a 'hole' in it that contains the firing pin impression

  Returns:
    drag_mask: A 2D numpy array mask representing the area of the firing pin drag. If no firing pin drag is found, the mask is all zeros
    points: A list of two coordinates representing the direction of the firing pin drag. The direction is starting at the first coordinate towards the second coordinate at index 1. If no firing pin drag is found this is returned as None
  '''

  center = seg.clear_border(1 - bf_mask) # take the inverse mask of the breach face and remove borders to get area within breach face.
  distance = ndi.distance_transform_edt(center)
  coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=center, num_peaks=2)
  # out = seg.slic(distance, n_segments=2, mask=center, channel_axis=None, max_num_iter=100, min_size_factor=0.1, compactness=1, max_size_factor=10)

  if len(coords) > 1:
    p1 = coords[0]
    p2 = coords[1]
    weight = (distance[p1[0]][p1[1]]/distance[p2[0]][p2[1]]) # Weight factor based on the size of each blob (represented by the height of each peak in distance feature). This factor could be improved to be more robust
    labels = np.zeros(distance.shape)
    for i in range(distance.shape[0]):
      for j in range(distance.shape[1]):
        if center[i][j]:
          d1 = np.linalg.norm(np.array([i, j]) - np.array(p1))
          d2 = np.linalg.norm(np.array([i, j]) - np.array(p2))
          
          if d1 < d2 * weight: # use weighting to make higher peak a larger area
            labels[i][j] = 1 # points closer to p1
          else:
            labels[i][j] = 2 # points closer to p2

    areas, counts = seperate_multiclass_mask(labels, sort_areas=False)
    drag = areas[np.argmin(counts)]
    if np.argmin(counts) == 0: # This means firing pin drag is centered on point 1
      # swap p1 and p2 since we will assume firing pin drag direction is from p1 to p2
      p1 = coords[1]
      p2 = coords[0]

  else:
    # If only 1 peak is found, that means there is not a significant second blob, hence a sufficiently large firing pin drag is not present. Return a mask of zeros
    drag = np.zeros(distance.shape)

  if len(coords) > 1:
    return drag, [p1, p2]
  else:
    return drag, None