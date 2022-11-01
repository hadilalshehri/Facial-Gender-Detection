#Importing the required libraries
import pandas as pd
import os
import cv2
import numpy as np
import time
import glob
import xgboost as xgb
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


##Combining the train and test data
# #Reading  the dataset 

def ReadDataSet(path_of_dataSet):
   Images_List=[]
   labels_List=[]

   # Reading ......
   for directory_path in glob.glob(path_of_dataSet):
       label = directory_path.split("\\")[-1]
       print(label)
       for img_path in glob.glob(os.path.join(directory_path, "*.jpg")): #jpg
           img = cv2.imread(img_path,0)# 0 for indicate the  gray scale   
           Images_List.append(img)
           labels_List.append(label)
   # Label Encode      
   le = preprocessing.LabelEncoder()
   le.fit (labels_List)
   Encoded_labels_List = le.transform(labels_List)
   Images_List = np.array(Images_List , dtype='object')
   Encoded_labels_List = np.array(Encoded_labels_List)
   return Images_List , Encoded_labels_List

# SIFT Feature 
def SIFT(Images_List, thresh):

  t0 = time.time()


  def CalcFeatures(img, th):
    sift = cv2.xfeatures2d.SIFT_create(th)
    kp, des = sift.detectAndCompute(img, None)
    return des
  
  '''
  All  Images list are passed through the CalcFeatures functions
   which returns the descriptors which are appended to the features list and 
  then stacked vertically in the form of a numpy array.
  '''

  features = []
  for img in Images_List:
    img_des = CalcFeatures(img, thresh)
    if img_des is not None:
      features.append(img_des)
  features = np.vstack(features)

  '''
  K-Means clustering is then performed on the feature array obtained 
  from the previous step. The centres obtained after clustering are 
  further used for bagging of features.
  '''

  k = 150
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
  flags = cv2.KMEANS_RANDOM_CENTERS
  compactness, labels, centres = cv2.kmeans(features, k, None, criteria, 10, flags)

  '''
  The bag_of_features function assigns the features which are similar
  to a specific cluster centre thus forming a Bag of Words approach.  
  '''

  def bag_of_features(features, centres, k = 500):
      vec = np.zeros((1, k))
      for i in range(features.shape[0]):
          feat = features[i]
          diff = np.tile(feat, (k, 1)) - centres
          dist = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
          idx_dist = dist.argsort()
          idx = idx_dist[0]
          vec[0][idx] += 1
      return vec

  vec = []
  for img in Images_List:
    img_des = CalcFeatures(img, thresh)
    if img_des is not None:
      img_vec = bag_of_features(img_des, centres, k)
      vec.append(img_vec)
  vec = np.vstack(vec)

  t1 = time.time()
  
  return vec, (t1-t0)

def SVM_Classifier(DataSet,Label_List):
     # Split to Train and Test
     X_train, X_test, y_train, y_test = train_test_split(DataSet,Label_List,test_size=.2)
                             
     # define support vector classifier
     svm = SVC(kernel='poly',probability=True, gamma = 0.6)#kernel='linear', probability=True,  C = 1,gamma= 'auto'
     svm.fit(X_train,y_train)
     y_pred = svm.predict(X_test)

     # calculate accuracy, specificity, sensitive and F1
     accuracy = accuracy_score(y_test, y_pred)
    #  specificity = recall_score(y_test, y_pred,average= None)
    #  sensitive = recall_score(np.logical_not(y_test) , np.logical_not(y_pred),average= None)
    #  F1 = f1_score(y_test, y_pred, average= None)
    #  report = classification_report(y_test, y_pred,zero_division =1, target_names=['Female', 'Male'])#,'Fear', 'Happy','Neutral','Sad','Suprise'
     conf_mat = confusion_matrix(y_test, y_pred)
     
    #  # Calculate AUC
    #  probabilities = svm.predict_proba(X_test)
    #  y_proba = probabilities[:, 1]
    #  # calculate false positive rate and true positive rate at different thresholds
    #  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)
    #  AUC = auc(false_positive_rate, true_positive_rate)
     

     return accuracy*100,conf_mat #specificity,sensitive,F1,report,false_positive_rate,true_positive_rate,AUC

def XgBoost_Classifier(DataSet,Label_List):
    # Split to Train and Test
     X_train, X_test, y_train, y_test = train_test_split(DataSet,Label_List,test_size=.2)

     model = xgb.XGBClassifier()
     model.fit(X_train,y_train)
     y_pred = model.predict(X_test)
     # Caculate accuracy,specificity,sensitive,F1,report
     accuracy= accuracy_score( y_pred,y_test)
     conf_mat = confusion_matrix(y_test, y_pred)
    #  specificity = recall_score(y_test, y_pred,average= None)
    #  sensitive = recall_score(np.logical_not(y_test) , np.logical_not(y_pred),average= None)
    #  F1 = f1_score(y_test, y_pred, average= None)
    #  report = classification_report(y_test, y_pred,zero_division =1, target_names=['Femal','male'])
     
    #  #AUC
    #  probabilities =model.predict_proba(X_test)
    #  y_proba = probabilities[:, 1]
    #  # calculate false positive rate and true positive rate at different thresholds
    #  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)
    #  AUC = auc(false_positive_rate, true_positive_rate)
    
     return accuracy*100,conf_mat #specificity,sensitive,F1,report ,false_positive_rate,true_positive_rate, AUC

# Body
Image_List,Label_list =ReadDataSet(r"Subset/*")

accuracy = []
timer = []
threshold = [] 
for i in range(5,26,5): #SVM  15,26,5 
  print('\nCalculating for a threshold of {}'.format(i))
  feature, time_consuming  = SIFT(Image_List,i)
  data= SVM_Classifier(feature, Label_list)
  accuracy.append(data[0])
  conf_mat = data[1]
  threshold.append(i)
  timer.append(time_consuming)
  
  print('\ngamma= {}\nAccuracy = {}\nTime taken = {} sec\nConfusion matrix :\n{}'.format(data[0],time_consuming,data[1]))

df = pd.DataFrame(columns=['threshold','accurecy'])
values = [threshold, accuracy]
for index,j in zip(df.columns,values):
    df[index] = j

df.to_excel('SVM Result.xlsx')
print("DataFrame is exported successfully to 'Feature Extraction Feelings.xlsx' Excel File.") 