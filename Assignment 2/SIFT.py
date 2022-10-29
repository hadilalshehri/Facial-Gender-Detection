#Importing the required libraries
import os
import cv2
import glob
import numpy as np
#i##mport vlfeat
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,f1_score,recall_score,classification_report

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

def SIFT_Feature(Images,thresh,no_sample):
    SIFT_DES = []
    vocab_Size = len(Images)/no_sample
    sift = cv2.xfeatures2d.SIFT_create(thresh)
    for img in Images:
    # Calculate the key points 
        keypoints, descriptors  = sift.detectAndCompute(img, None)
        #DES_Sample = descriptors[np.random.randint(descriptors.shape[0],size = no_sample)]
        SIFT_DES.append(descriptors) # AUTHOR ONLY USE DESC AS FEATURE 
    
    Criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    _,_,centers  = cv2.kmeans(SIFT_DES ,vocab_Size,None,Criteria,10,cv2.KMEANS_PP_CENTERS)
    return SIFT_DES, centers 

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
Image_List,Label_list =ReadDataSet(r"Subset/*")
thresh= [20,5,10] # number of key points 
SIFT_Feature , vocab= SIFT_Feature(Image_List,thresh[1],20)
vec = bag_of_features(SIFT_Feature , vocab, k=500)
SIFT_Matrix =vec# SIFT_Feature#np.array(SIFT_Feature)
print(SIFT_Matrix)


#______________________XgBoost_Classifier___________________________
#Classifier SIFT Feature using XgBoost
#SIFT_accuracy ,SIFT_Specificity,SIFT_Sensitive, SIFT_F1, SIFT_Report,SIFT_FBR,SIFT_TBR,AUC = XgBoost_Classifier(SIFT_Matrix,Label_list)
#PrintValue('SIFT Classification Using XgBoost',SIFT_accuracy ,SIFT_F1,SIFT_Specificity,SIFT_Sensitive, SIFT_Report)


#______________________SVM_Classifier_______________________________
#Classifier SIFT Feature using SVM
SVM_SIFT_accuracy ,SVM_SIFT_Specificity,SVM_SIFT_Sensitive, SVM_SIFT_F1, SVM_SIFT_Report,SVM_SIFT_FBR,SVM_SIFT_TBR,SVM_AUC = SVM_Classifier(SIFT_Matrix,Label_list)
PrintValue('SIFT Classification Using SVM',SVM_SIFT_accuracy ,SVM_SIFT_F1,SVM_SIFT_Specificity,SVM_SIFT_Sensitive, SVM_SIFT_Report)


#_____________________________Plot___________________________________
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
ax[0][0].plot(SIFT_FBR,SIFT_TBR,label='AUC = {:0.2f}'.format(AUC))
ax[0][0].plot([0,1], [0,1], ls='--')    
ax[0][0].set_title('SIFT XgBoost Classifier')
plotnew(ax[0][0])
ax[0][1].plot(SVM_SIFT_FBR,SVM_SIFT_TBR,label='AUC = {:0.2f}'.format(SVM_AUC))
ax[0][1].plot([0,1], [0,1], ls='--') 
ax[0][1].set_title('SIFT SVM Classifier')
plotnew(ax[0][1])
"""
ax[1][0].plot(SVM_GLCM_FBR,SVM_GLCM_TBR,label='AUC = {:0.2f}'.format(SVM_G_AUC))
ax[1][0].plot([0,1], [0,1], ls='--') 
ax[1][0].set_title('GLCM SVM Classifier')
plotnew(ax[1][0])

ax[1][1].plot(SVM_SIFT_FBR,SVM_SIFT_TBR,label='AUC = {:0.2f}'.format(SVM_AUC))
ax[1][1].plot([0,1], [0,1], ls='--') 
ax[1][1].set_title('SIFT SVM Classifier')
plotnew(ax[1][1])
"""
plt.show()

"""
# Store GLCM & SIFT on one Data Frame
df = pd.DataFrame(columns=['label','Key Point','Dis'])
Features = [Label_list, SIFT_KeyPoint,SIFT_DESCRIPTOR]
for i,j in zip(df.columns,Features):
    df[i] = j

df.to_excel('Feature Extraction SIFT.xlsx')
print("DataFrame is exported successfully to 'Feature Extraction Feelings.xlsx' Excel File.")
"""
