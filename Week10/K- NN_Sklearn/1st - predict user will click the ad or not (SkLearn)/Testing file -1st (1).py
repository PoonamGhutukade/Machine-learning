#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import libraries
import pandas as pd

#libarry for feature scaling
from sklearn.preprocessing import StandardScaler

# #confusion matix
from sklearn import metrics
# to creating and reading pickle file
import pickle 
import os, sys
import csv

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# from util import Util_class as obj_util
import importlib.util


# In[5]:


# importing template file 
spec = importlib.util.spec_from_file_location("Util_class", "/home/admin1/PycharmProjects/Machine-Learning/Week10/Util/util.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
obj_util = foo.Util_class()


# In[6]:


# load dataset
test_dataset = pd.read_csv ("CSV_files/test_file.csv")
print("Dataset has {} rows and {} Columns".format(test_dataset.shape[0],test_dataset.shape[1])) 


# In[7]:


#spliting data 
x_test = test_dataset.iloc[:,[2,3]].values
y_test = test_dataset.iloc[:,4].values  

print("x_test :",x_test.shape,"& y_test:",y_test.shape)


# In[8]:


#feature scalling (here data will be converted into float)
def feature_scalling(x_test,y_test):
    sc_x = StandardScaler()
    x_test = sc_x.fit_transform(x_test)
    return sc_x, x_test
    
sc_x, x_test = feature_scalling(x_test,y_test)


# In[11]:


#load model
file1 = open('KNN.pkl', 'rb')
classifier = pickle.load(file1)


# In[16]:


def final_tetsing():
    # y prediction for test data
    y_pre = obj_util.y_prediction(x_test, classifier)
#     print("\n\n y_prediction:",y_pre)
    
    # calculate accuracy
    accuracy_score,average_precision,auc=obj_util.accuracy(y_pre,y_test)
    print('\n\nAverage accuracy_score:' , accuracy_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Average Roc-AUC: %.3f' % auc)

    # show confusion matrix
    cm = obj_util.confusion_matrix(y_test, y_pre)
    print("\n\nConfusion Matrix:\n",cm)
    
#     print("\n\nConfusion Matrix:\n",metrics.confusion_matrix(y_test, y_pre))
    obj_util.visualization(x_test,y_test, classifier, "KNN Classification(Testing set)", 
                           "Age", "Estimate Salary")
    
# call function
final_tetsing()


# In[ ]:




