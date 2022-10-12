import pandas as pd
import numpy as np
import glob
import os    
import cv2
import matplotlib.pyplot as plt
import xgboost as xgb
from skimage import feature, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,f1_score,recall_score,classification_report

# This code have been refernced to 
# GLCM Code
# https://github.com/Madhu87/Feature-Extraction-using-GLCM
# SVM Code
#https://rpubs.com/Sharon_1684/454441
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def ReadDataSet(path_of_dataSet):
  # Reading the DataSet and store it on  List 
   Images_List=[]
   labels_List=[]

   # Reading Data SET
   for directory_path in glob.glob(path_of_dataSet):
       label = directory_path.split("\\")[-1]
       print(label)
       for img_path in glob.glob(os.path.join(directory_path, "*.jpg")): #jpg
           img = cv2.imread(img_path,0)# ,0 for indicate the  gray scale   
           Images_List.append(img)
           labels_List.append(label)
        
   le = preprocessing.LabelEncoder()
   le.fit (labels_List)
   Encoded_labels_List = le.transform(labels_List)
   Images_List = np.array(Images_List , dtype='object')
   Encoded_labels_List = np.array(Encoded_labels_List)
   return Images_List , Encoded_labels_List

def LBP_ExtractFeature(Images_List):
    #
    HistList = []
    for matrix in Images_List:
        numPoints = 24
        radius = 3
        lbp = feature.local_binary_pattern(matrix, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))
        """
     # Normalize the Histogram
        eps = 1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        """
        HistList.append(hist)

    return HistList 

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
       image = img_as_ubyte(matrix)# Becomes 8-bit unsigned integer
       GLCM_Image = graycomatrix(image, [1,2,3], [0, np.pi/4, np.pi/2, 3*np.pi/4],normed=False, symmetric=False)

                                          
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

def GLCM_Feauter_Extraction_Vector(Images_List):
   Feature =[]

# Calculating the GLCM feature for each image  
   for matrix in Images_List:
       image = img_as_ubyte(matrix)# Becomes 8-bit unsigned integer
       GLCM_Image = graycomatrix(image, [1,2,3], [0, np.pi/4, np.pi/2, 3*np.pi/4],normed=False, symmetric=False)

       contrast = graycoprops(GLCM_Image,'contrast')[0][0]     
       dissimilarity = graycoprops(GLCM_Image,'dissimilarity')[0][0]    
       homogeneity = graycoprops(GLCM_Image,'homogeneity')[0][0]   
       energy = graycoprops(GLCM_Image,'energy')[0][0]  
       correlation = graycoprops(GLCM_Image,'correlation')[0][0]  
       asm = graycoprops(GLCM_Image,'ASM')[0][0]  
       temp = [contrast,dissimilarity,homogeneity,energy,correlation,asm]
       Feature.append(temp)

   return Feature

def SVM_Classifier(DataSet,Label_List):
     # Split to Train and Test
     X_train, X_test, y_train, y_test = train_test_split(DataSet,Label_List,test_size=.2)
                             
     # define support vector classifier
     svm = SVC(kernel='linear', probability=True,  C = 1,gamma= 'auto')
   #  svm = SVC(kernel='rbf', C=1e3, gamma=0.1)
     # fit model
     svm.fit(X_train,y_train)
     # generate predictions
     y_pred = svm.predict(X_test)

     # calculate accuracy, specificity, sensitive and F1
     accuracy = accuracy_score(y_test, y_pred)
     specificity = recall_score(y_test, y_pred,average= None)
     sensitive = recall_score(np.logical_not(y_test) , np.logical_not(y_pred),average= None)
     F1 = f1_score(y_test, y_pred, average= None)
     report = classification_report(y_test, y_pred,zero_division =1, target_names=['Female', 'Male'])#,'Fear', 'Happy','Neutral','Sad','Suprise'
     
     
     # Calculate AUC
     probabilities = svm.predict_proba(X_test)
     y_proba = probabilities[:, 1]
     # calculate false positive rate and true positive rate at different thresholds
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)
     AUC = auc(false_positive_rate, true_positive_rate)
     

     return accuracy,specificity,sensitive,F1,report,false_positive_rate,true_positive_rate,AUC

def XgBoost_Classifier(DataSet,Label_List):
    # Split to Train and Test
     X_train, X_test, y_train, y_test = train_test_split(DataSet,Label_List,test_size=.2)

     model = xgb.XGBClassifier()
     model.fit(X_train,y_train)
     y_pred = model.predict(X_test)
     # Caculate accuracy,specificity,sensitive,F1,report
     accuracy= accuracy_score( y_pred,y_test)
     specificity = recall_score(y_test, y_pred,average= None)
     sensitive = recall_score(np.logical_not(y_test) , np.logical_not(y_pred),average= None)
     F1 = f1_score(y_test, y_pred, average= None)
     report = classification_report(y_test, y_pred,zero_division =1, target_names=['Femal','male'])
     
     #AUC
     probabilities =model.predict_proba(X_test)
     y_proba = probabilities[:, 1]
     # calculate false positive rate and true positive rate at different thresholds
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)
     AUC = auc(false_positive_rate, true_positive_rate)
    
     return accuracy,specificity,sensitive,F1,report ,false_positive_rate,true_positive_rate, AUC

def plotnew(ax):
    ax.set_xlabel('True Positive Rate')
    ax.set_ylabel('False Positive Rate')

def PrintValue(method,accuracy ,F1,Specificity,Sensitive,Report):
    print('--- ',method,' ---')
    print('Model accuracy is: ', accuracy*100,'%')
    print('F1 score is',F1)
    print('Specificity is ',Specificity)
    print('Sensitive is ',Sensitive)
    print('--------- ',method,' Report------------')
    print(Report)

#______________________MAIN___________________________
#Read Data SET IMAGES 
Image_List,Label_list =ReadDataSet(r"C:/Users/alshehriHa9/Documents/Master/Image processing/Gender/Subset/*")

#Extract Histigram Feature from LBP
Histogram_Feature_list = LBP_ExtractFeature(Image_List)
LBP_Matrix = np.array(Histogram_Feature_list,dtype = 'object')

# Extract Texture Featrue using GLCM
contrast_feature,dissimilarity_feature,homogeneity_feature,energy_feature,correlation_feature,ASM_feature = GLCM_Feauter_Extraction(Image_List)

# Concate GLCM and LBP Feature
GLCM_LIST = GLCM_Feauter_Extraction_Vector(Image_List)
glcm_Matrix = np.array(GLCM_LIST,dtype = 'object')
LBP_GLCM_Feature =np.hstack([glcm_Matrix,LBP_Matrix])

# Creating Data Frame to store GLCM feature
GLCM_df = pd.DataFrame(columns=['Contrast Feature(GLCM)','Dissimilarity Feature(GLCM)','Homogeneity Feature(GLCM)','Energy Feature(GLCM)','Correlation Feature(GLCM)','ASM Feature(GLCM)'])

Features = [contrast_feature,dissimilarity_feature,homogeneity_feature,energy_feature,correlation_feature,ASM_feature]
for i,j in zip(GLCM_df.columns,Features):
    GLCM_df[i] = j

# Store GLCM & LBP on one Data Frame
df = pd.DataFrame(columns=['label','Histogram (LBP)','Contrast Feature(GLCM)','Dissimilarity Feature(GLCM)','Homogeneity Feature(GLCM)','Energy Feature(GLCM)','Correlation Feature(GLCM)','ASM Feature(GLCM)'])
Features = [Label_list,Histogram_Feature_list,contrast_feature,dissimilarity_feature,homogeneity_feature,energy_feature,correlation_feature,ASM_feature]
for i,j in zip(df.columns,Features):
    df[i] = j


#______________________XgBoost_Classifier___________________________
#Classifier LBP Feature using XgBoost
LBP_accuracy ,LBP_Specificity,LBP_Sensitive, LBP_F1, LBP_Report,LBP_FBR,LBP_TBR,AUC = XgBoost_Classifier(LBP_Matrix,Label_list)
PrintValue('LBP Classification Using XgBoost',LBP_accuracy ,LBP_F1,LBP_Specificity,LBP_Sensitive, LBP_Report)

#Classifier GLCM Feature using XgBoost
GLCM_accuracy,GLCM_Specificity,GLCM_Sensitive, GLCM_F1,GLCM_Report,GLCM_FBR,GLCM_TBR,G_AUC = XgBoost_Classifier(GLCM_df,Label_list)
PrintValue('GLCM Classification Using XgBoost',GLCM_accuracy,GLCM_F1,GLCM_Specificity,GLCM_Sensitive,GLCM_Report)

#Classifier LBP  & GLCM  using XgBoost
LBP_GLCM_accuracy,LBP_GLCM_Specificity,LBP_GLCM_Sensitive, LBP_GLCM_F1,LBP_GLCM_Report,LBP_GLCM_FBR,LBP_GLCM_TBR,LBP_G_AUC = XgBoost_Classifier(LBP_GLCM_Feature,Label_list)
PrintValue('LBP and GLCM  Classification Using XgBoost',LBP_GLCM_accuracy,LBP_GLCM_F1,LBP_GLCM_Specificity,LBP_GLCM_Sensitive,LBP_GLCM_Report)


#______________________SVM_Classifier_______________________________
#Classifier LBP Feature using SVM
SVM_LBP_accuracy ,SVM_LBP_Specificity,SVM_LBP_Sensitive, SVM_LBP_F1, SVM_LBP_Report,SVM_LBP_FBR,SVM_LBP_TBR,SVM_AUC = SVM_Classifier(LBP_Matrix,Label_list)
PrintValue('LBP Classification Using SVM',SVM_LBP_accuracy ,SVM_LBP_F1,SVM_LBP_Specificity,SVM_LBP_Sensitive, SVM_LBP_Report)

#Classifier GLCM Feature using SVM
SVM_GLCM_accuracy,SVM_GLCM_Specificity,SVM_GLCM_Sensitive, SVM_GLCM_F1,SVM_GLCM_Report,SVM_GLCM_FBR,SVM_GLCM_TBR,SVM_G_AUC = SVM_Classifier(GLCM_df,Label_list)
PrintValue('GLCM Classification Using SVM',SVM_GLCM_accuracy,SVM_GLCM_F1,SVM_GLCM_Specificity,SVM_GLCM_Sensitive,SVM_GLCM_Report)

#Classifier LBP  & GLCM using SVM
G_SVM_LBP_accuracy ,G_SVM_LBP_Specificity,G_SVM_LBP_Sensitive, G_SVM_LBP_F1, G_SVM_LBP_Report,vSVM_LBP_FBR,G_SVM_LBP_TBR,G_SVM_AUC = SVM_Classifier(LBP_GLCM_Feature,Label_list)
PrintValue('LBP and GLCM Classification Using SVM',G_SVM_LBP_accuracy ,G_SVM_LBP_F1,G_SVM_LBP_Specificity,G_SVM_LBP_Sensitive, G_SVM_LBP_Report)



#_____________________________Plot___________________________________
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
ax[0][0].plot(GLCM_FBR,GLCM_TBR,label='AUC = {:0.2f}'.format(G_AUC))
ax[0][0].plot([0,1], [0,1], ls='--')    
ax[0][0].set_title('GLCM XgBoost Classifier')
plotnew(ax[0][0])
ax[0][1].plot(LBP_FBR,LBP_TBR,label='AUC = {:0.2f}'.format(AUC))
ax[0][1].plot([0,1], [0,1], ls='--') 
ax[0][1].set_title('LBP XgBoost Classifier')
plotnew(ax[0][1])

ax[1][0].plot(SVM_GLCM_FBR,SVM_GLCM_TBR,label='AUC = {:0.2f}'.format(SVM_G_AUC))
ax[1][0].plot([0,1], [0,1], ls='--') 
ax[1][0].set_title('GLCM SVM Classifier')
plotnew(ax[1][0])

ax[1][1].plot(SVM_LBP_FBR,SVM_LBP_TBR,label='AUC = {:0.2f}'.format(SVM_AUC))
ax[1][1].plot([0,1], [0,1], ls='--') 
ax[1][1].set_title('LBP SVM Classifier')
plotnew(ax[1][1])

plt.show()


# Store GLCM & LBP on one Data Frame
df = pd.DataFrame(columns=['label','Histogram (LBP)','Contrast Feature(GLCM)','Dissimilarity Feature(GLCM)','Homogeneity Feature(GLCM)','Energy Feature(GLCM)','Correlation Feature(GLCM)','ASM Feature(GLCM)'])
Features = [Label_list,Histogram_Feature_list,contrast_feature,dissimilarity_feature,homogeneity_feature,energy_feature,correlation_feature,ASM_feature]
for i,j in zip(df.columns,Features):
    df[i] = j

df.to_excel('Feature Extraction Gender.xlsx')
print("DataFrame is exported successfully to 'Feature Extraction Feelings.xlsx' Excel File.")

