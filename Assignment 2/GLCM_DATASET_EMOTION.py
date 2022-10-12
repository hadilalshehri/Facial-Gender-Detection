import pandas as pd
import numpy as np
import glob
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os    
import cv2

# This code have been refernced to 
# https://github.com/Madhu87/Feature-Extraction-using-GLCM

def ReadDataSet(path_of_dataSet):
# Reading the DataSet and store it on  List 
   Images_List=[]
   labels_List=[]
   #path_of_images = r"C:/Users/alshehriHa9/Documents/Master/Image processing/Python code IMP/Assignment 2/Face felling/Training/Sub Training/*"

# Reading Data SET
   for directory_path in glob.glob(path_of_dataSet):
       label = directory_path.split("\\")[-1]
       print(label)
       for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
           img = cv2.imread(img_path) 
           Images_List.append(img)
           labels_List.append(label)
        
   Images_List = np.array(Images_List)
   labels_List = np.array(labels_List)
   return Images_List , labels_List


def GLCM_Feauter_Extraction(Images_List):
# GLCM Properties 
   CF =[]
   DF =[]
   HF =[]
   EF =[]
   COR = []
   ASM = []

# Calculating the GLCM feature for each image  
   for matrix in Images_List:
       gray = color.rgb2gray(matrix)
       image = img_as_ubyte(gray)# Becomes 8-bit unsigned integer
       GLCM_Image = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],normed=False, symmetric=False)

                                          
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
    
       CF.append(np.mean(contrast_feature(GLCM_Image)))
       DF.append(np.mean(dissimilarity_feature(GLCM_Image)))
       HF.append(np.mean(homogeneity_feature(GLCM_Image)))
       EF.append(np.mean(energy_feature(GLCM_Image)))
       COR.append(np.mean(correlation_feature(GLCM_Image)))
       ASM.append(np.mean(asm_feature(GLCM_Image)))
   return CF,DF,HF,EF,COR,ASM

#Read Data SET IMAGES 
train_images,train_labels =ReadDataSet(r"C:/Users/alshehriHa9/Documents/Master/Image processing/Python code IMP/Assignment 2/Face Feelings/Training/Sub Training/*")

# Extract Texture Featrue using GLCM
contrast_feature,dissimilarity_feature,homogeneity_feature,energy_feature,correlation_feature,ASM_feature = GLCM_Feauter_Extraction(train_images)

# creating Data Frame to store the Image feature and Label
df = pd.DataFrame(columns=['label','Contrast Feature','Dissimilarity Feature','Homogeneity Feature','Energy Feature','Correlation Feature','ASM Feature'])

# Add Label & feautre to Data Frame
Features = [train_labels,contrast_feature,dissimilarity_feature,homogeneity_feature,energy_feature,correlation_feature,ASM_feature]
for i,j in zip(df.columns,Features):
    df[i] = j

df.set_index(train_labels)
df.index.name = 'Image ID'

# Store image Feature to Excel
df.to_excel('Feature Extraction Feelings.xlsx')
print("DataFrame is exported successfully to 'Feature Extraction Feelings.xlsx' Excel File.")
