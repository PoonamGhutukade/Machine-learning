#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Classification
	Logistic Regression
Build a machine learning model to predict user will click the ad or not based on his experience 
and estimated salary for a given dataset.

"""


# In[2]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  
import pandas as pd
import seaborn as sb
#imputer to handle missing data 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
# handle categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

#Classification library
from sklearn.linear_model import LogisticRegression
#confusion matix
from sklearn.metrics import confusion_matrix 
#visualisation
from matplotlib.colors import ListedColormap
# calculate accuracy
from sklearn import metrics
#o check accuracy
from sklearn.metrics import accuracy_score
# to check accuracy
from sklearn.metrics import *

import pickle 
import os, sys
import csv

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# from util import Util_class as obj_util
import importlib.util


# In[3]:



# importing template file 
spec = importlib.util.spec_from_file_location("Util_class", "/home/admin1/PycharmProjects/Machine-Learning/Week10/Util/util.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of templates class
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


type(dataset.Gender)


# In[8]:


# descibe the dataset
dataset.describe().T


# In[9]:


# handling missing data if nessesary
"""
if missing values are present
imputer = Imputer(missing_values=0, axis=0)
imputer = imputer.fit(x_data[:, 3:16])
"""
dataset.isnull().sum()


# In[10]:


# check for duplicate values
dataset.duplicated().sum()


# In[11]:


#Check categpries for categorical data
dataset.Gender[:5]


# In[12]:


#Display heatmap to show correlation between diff variables
corr = dataset.corr()
sb.heatmap(corr)


# In[13]:


# create directory to store csv files
os.mkdir("CSV_files")


# In[14]:


#split dataset into train, test and cross validation also save csv files
obj_util.splitdata(dataset, 0.30, 0.40,"CSV_files" )


# In[15]:


# load train dataset
train_dataset = pd.read_csv ("CSV_files/train_file.csv")
print("Train Dataset has {} rows and {} Columns".format(train_dataset.shape[0],train_dataset.shape[1])) 
# load dataset for Cross Validation
CV_dataset = pd.read_csv ("CSV_files/CValidation_file.csv")
print("Cross validation Dataset has {} rows and {} Columns".format(CV_dataset.shape[0],CV_dataset.shape[1])) 


# In[16]:


#data Preprocessing
train_dataset.info()


# In[17]:


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


# In[18]:


type(x_train)


# In[19]:


#feature scalling (here data will be converted into float)
def feature_scalling(x_train,x_crossval):
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    
    sc_x_cv = StandardScaler()
    x_crossval = sc_x.fit_transform(x_crossval)
    return sc_x, x_train,sc_x_cv, x_crossval
    
sc_x, x_train,sc_x_cv, x_crossval = feature_scalling(x_train,x_crossval)


# In[20]:


# x_train = pd.Series(x_train)


# In[21]:


class LogisticReg():
    
    def create_module(self,x_train,y_train):
        # fitting LogisticRegression to the training set
        classifier = LogisticRegression()
        classifier.fit(x_train,y_train)
        return classifier    
    

def main():
    #class obj created
    obj  = LogisticReg()

    classifier = obj.create_module(x_train,y_train)
    print("\nModule created")
    print("regression object",type(classifier))


    y_pre = obj_util.y_prediction(x_train, classifier)
    print("\n\n y_prediction:",y_pre)
    print(y_pre.shape)
    
    accuracy_score,average_precision,auc=obj_util.accuracy(y_pre,y_train)
    
    print('\n\nAverage accuracy_score:' , accuracy_score)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    print('Average Roc-AUC: %.3f' % auc)

    
    print("\n\nConfusion Matrix:\n",metrics.confusion_matrix(y_train, y_pre))
    obj_util.visualization(x_train,y_train, classifier, "Logistic Regression(Training set)", 
                           "Age", "Estimate Salary")
    
    obj_util.create_piklefile(classifier,'LogisticRegression.pkl' )
    print("\nPikle file created")


if __name__ == '__main__':
    main()


# In[22]:


# cross validation        
def Cross_validation():
    file1 = open('LogisticRegression.pkl', 'rb')
    classifier1 = pickle.load(file1)

    # y_prediction ( cross validation) 
    y_predicted1 = obj_util.y_prediction(x_crossval, classifier1)
    print("\n\n y_prediction:",y_predicted1)
    
    print(y_crossval.shape, y_predicted1.shape)
    accuracy_score,average_precision,auc=obj_util.accuracy(y_predicted1, y_crossval)
    
    print('\n\nAverage accuracy_score:' , accuracy_score)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    print('Average Roc-AUC: %.3f' % auc)

    
    print("\n\nConfusion Matrix:\n",metrics.confusion_matrix(y_crossval, y_predicted1))
    
    obj_util.visualization(x_crossval, y_crossval, classifier1, "Logistic Regression(Training set)", "Age", "Estimate Salary")
    
    
    

Cross_validation()


# In[ ]:




