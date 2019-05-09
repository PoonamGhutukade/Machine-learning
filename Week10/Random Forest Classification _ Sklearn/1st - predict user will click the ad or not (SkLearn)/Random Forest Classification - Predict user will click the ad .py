#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Random Forest Classification
1.  Build a machine learning model to predict user will click the ad or not based on his experience and 
estimated salary for a given dataset.

"""


# In[2]:


#import libraries
import pandas as pd
import seaborn as sb

#libarry for feature scaling
from sklearn.preprocessing import StandardScaler
# #Classification library
from sklearn.ensemble import RandomForestClassifier

import pickle 
import os, sys
import csv

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import importlib.util


# In[3]:


# importing template file 
spec = importlib.util.spec_from_file_location("Util_class", "/home/admin1/PycharmProjects/Machine-Learning/Week10/Util/util.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
obj_util = foo.Util_class()


# In[4]:


# load dataset
dataset_original = pd.read_csv ("Social_Network_Ads.csv")
dataset = dataset_original
dataset.head()


# In[5]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[6]:


# check dataset information
dataset.info()


# In[7]:


# descibe the dataset
dataset.describe().T


# In[8]:


# handling missing data if nessesary
"""
if missing values are present
imputer = Imputer(missing_values=0, axis=0)
imputer = imputer.fit(x_data[:, 3:16])
"""
dataset.isnull().sum()


# In[9]:


# check for duplicate values
dataset.duplicated().sum()


# In[10]:


#categorical data
type(dataset.Gender)


# In[11]:


#Check categpries for categorical data
dataset.Gender[:5]


# In[12]:


#Display heatmap to show correlation between diff variables
corr = dataset.corr()
sb.heatmap(corr)


# In[13]:


# check skewness for target variable
sb.distplot(dataset['Purchased'])
print ("Skewness of y is {}".format(dataset['Purchased'].skew()))


# In[14]:


print (corr['Purchased'].sort_values(ascending=False)[:10], '\n') #top 10 values
print ('----------------------')
print (corr['Purchased'].sort_values(ascending=False)[-5:]) #last 5 values`


# In[15]:


# create directory to store csv files
# os.mkdir("CSV_files")


# In[16]:


#split dataset into train, test and cross validation also save csv files
obj_util.splitdata(dataset, 0.30, 0.40,"CSV_files" )


# In[17]:


# load train dataset
train_dataset = pd.read_csv ("CSV_files/train_file.csv")
print("Train Dataset has {} rows and {} Columns".format(train_dataset.shape[0],train_dataset.shape[1])) 
# load dataset for Cross Validation
CV_dataset = pd.read_csv ("CSV_files/CValidation_file.csv")
print("Cross validation Dataset has {} rows and {} Columns".format(CV_dataset.shape[0],CV_dataset.shape[1])) 


# In[18]:


#data Preprocessing
train_dataset.info()


# In[19]:


# seperate fetures and label

# features -> age and estimated salary
x_train = train_dataset.iloc[:,[2,3]].values
# label -> purchased
y_train = train_dataset.iloc[:,4].values  

# Dont reshape any variable it gives error for visualisation "IndexError: too many indices for array"
# y_train = y_train.reshape(-1,1)
print("x_train :",x_train.shape,"& y_train:",y_train.shape)

#for cross validation
# features -> age and estimated salary
x_crossval = CV_dataset.iloc[:,[2,3]].values
# label -> purchased
y_crossval = CV_dataset.iloc[:,4].values  

print("x_cv :",x_crossval.shape,"& y_cv:",y_crossval.shape)


# In[20]:


#feature scalling (here data will be converted into float)
def feature_scalling(x_train,x_crossval):
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    
    sc_x_cv = StandardScaler()
    x_crossval = sc_x.fit_transform(x_crossval)
    return sc_x, x_train,sc_x_cv, x_crossval
    
sc_x, x_train,sc_x_cv, x_crossval = feature_scalling(x_train,x_crossval)


# In[21]:


class Random_forest():
    
    # create random forest Model
    def create_module(self,x_train,y_train):
        # fitting KNN Classification to the training set
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(x_train,y_train)
        return classifier
    

def main():
    #class obj created
    obj  = Random_forest()

     # create random forest Model
    classifier = obj.create_module(x_train,y_train)
    print("\nModule created")
    print("classifier object",classifier)

    # y prediction
    y_pre = obj_util.y_prediction(x_train, classifier)
    print("\n\n y_prediction:",y_pre)
    print(y_pre.shape)
    
    # calculate accuracy
    accuracy_score,average_precision,auc=obj_util.accuracy(y_pre,y_train)
    print('\n\nAverage accuracy_score:' , accuracy_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Average Roc-AUC: %.3f' % auc)

    # show confusion matrix
    cm = obj_util.confusion_matrix(y_train,y_pre)
    print("\n\nConfusion Matrix:\n",cm)
    
    # data visualisation
    obj_util.visualization(x_train,y_train, classifier, "Random_forest Classification(Training set)", 
                           "Age", "Estimate Salary")
    
    # create pickle file
    obj_util.create_piklefile(classifier,'Random_forest.pkl' )
    print("\nPikle file created")


if __name__ == '__main__':
    main()


# In[22]:


# cross validation        
def Cross_validation():
    # load pickle file 
    file1 = open('Random_forest.pkl', 'rb')
    classifier1 = pickle.load(file1)

    # y_prediction ( cross validation) 
    y_predicted1 = obj_util.y_prediction(x_crossval, classifier1)
    print("\n\n y_prediction:",y_predicted1)
    
#     print(y_crossval.shape, y_predicted1.shape)
    # calculate accuracy
    accuracy_score,average_precision,auc=obj_util.accuracy(y_predicted1, y_crossval)
    print('\n\nAverage accuracy_score:' , accuracy_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Average Roc-AUC: %.3f' % auc)

    # show confusion matrix
    cm = obj_util.confusion_matrix(y_crossval, y_predicted1)
    print("\n\nConfusion Matrix:\n",cm)
#     print("\n\nConfusion Matrix:\n",metrics.confusion_matrix(y_crossval, y_predicted1))
    
    obj_util.visualization(x_crossval, y_crossval, classifier1, "Random_forest(Cross_validation set)", "Age", "Estimate Salary")
    
    
Cross_validation()


# In[ ]:





# In[ ]:




