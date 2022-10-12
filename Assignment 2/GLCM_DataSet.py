import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os    
import cv2

# creating Data Frame to store the Image feature 
df = pd.DataFrame(columns=['Contrast Feature','Dissimilarity Feature','Homogeneity Feature','Energy Feature','Correlation Feature','ASM Feature'])

# Reading the DataSet and store it on  Matrix 
matrix1 = []
#path_of_images = r"C:/Users/alshehriHa9/Documents/Master/Image processing/Python code IMP/Assignment 2/Face felling/Training/Training/Angry"
path_of_images = r"C:/Users/alshehriHa9/Documents/Master/Image processing/Python code IMP/Assignment 2/data"

#path_of_images = r"Assignment 2/Dataset"
list_of_images = os.listdir(path_of_images)

for image in list_of_images:
    img = cv2.imread(os.path.join(path_of_images, image))
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)# Becomes 8-bit unsigned integer

#This step is similar to data compression, because the 8-bit image contains 256 gray levels, which will cause the calculation of the gray level co-occurrence matrix to be too large, so it is compressed into 16 levels and the gray levels are divided
  #  bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
  #  inds = np.digitize(image, bins)#Returns a matrix with the same size as the image, but the matrix element represents the interval position of the element in the image in the bins, less than 0 is 0, 0-16 is 1, and so on
# Calculating GLCM
   # max_value = inds.max()+1
    GLCM_Image = graycomatrix(image, #Numpy matrix for co-occurrence matrix calculation
                                  [1], #Distance
                                  [0, np.pi/4, np.pi/2, 3*np.pi/4], #Direction angle
                                 # levels=max_value, #Co-occurrence matrix order
                                  normed=False, symmetric=False)
#P[i,j,d,theta] returns a four-dimensional matrix, each dimension represents a different meaning
    matrix1.append(GLCM_Image)

# GLCM Properties 
CF =[]
DF =[]
HF =[]
EF =[]
COR = []
ASM = []

# Calculating the GLCM feature for each image  
for matrix in matrix1:
    #angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    #grayImg = img_as_ubyte(color.rgb2gray(matrix))
    #glcm = graycomatrix(grayImg,  distances=distances, angles=angles,symmetric=True,normed=True)

    def contrast_feature(matrix):
       contrast = graycoprops(matrix,'contrast')
       return  contrast

    def dissimilarity_feature(matrix):
       dissimilarity = graycoprops(matrix,'dissimilarity')
       return list(dissimilarity)

    def homogeneity_feature(matrix):
       homogeneity = graycoprops(matrix,'homogeneity')
       return  list(homogeneity)

    def energy_feature(matrix):
       energy = graycoprops(matrix,'energy')
       return list(energy)

    def correlation_feature(matrix):
       correlation = graycoprops(matrix,'correlation')
       return list(correlation)

    def asm_feature(matrix):
       asm = graycoprops(matrix,'ASM')
       return list(asm)
    
    CF.append(np.mean(contrast_feature(matrix)))
    DF.append(np.mean(dissimilarity_feature(matrix)))
    HF.append(np.mean(homogeneity_feature(matrix)))
    EF.append(np.mean(energy_feature(matrix)))
    COR.append(np.mean(correlation_feature(matrix)))
    ASM.append(np.mean(asm_feature(matrix)))


# Add feautre to Data Frame
Features = [CF,DF,HF,EF,COR,ASM]
for i,j in zip(df.columns,Features):
    df[i] = j

#for i in range(matrix1):
 #   df.index = ['Image',i]
#df.set_index(range(len(matrix1)))

df.index = ['Image1','Image2','Image3','Image4','Image5','Image6','Image7']

df.index.name = 'Image ID'
print(df)
# Store image Feature to Excel
df.to_excel('Feature Extraction.xlsx')
# save the excel
print("DataFrame is exported successfully to 'Feature Extraction.xlsx' Excel File.")
