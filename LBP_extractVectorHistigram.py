import cv2
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage import feature
import numpy as np
  
# define the parameters of the Local Binary Patterns

# extract the histogram of Local Binary Patterns
def LBP_ExtractFeature(grayImage):
    #
    numPoints = 24
    radius = 3
    lbp = feature.local_binary_pattern(grayImage, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3),
            range=(0, numPoints + 2))
  
# optionally normalize the histogram
    eps = 1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist 

img = cv2.imread('king.jpg')
gray = color.rgb2gray(img)
hist = LBP_ExtractFeature(gray)

print(hist)