import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy

rgbImg = io.imread('pic1.jpg')
grayImg = img_as_ubyte(color.rgb2gray(rgbImg))

distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['energy', 'homogeneity']

glcm = graycomatrix(grayImg, 
                    distances=distances, 
                    angles=angles,
                    symmetric=True,
                    normed=True)

feats = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
np.set_printoptions(precision=4)
print(feats)

