#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  
import pandas as pd

#imputer to handle missing data 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#o check accuracy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# to check accuracy
from sklearn.metrics import *
from sklearn import metrics
import pickle 

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import csv
# from util import Util_class as obj_util
import importlib.util


# In[2]:


# importing template file 
spec = importlib.util.spec_from_file_location("Util_class", "/home/admin1/PycharmProjects/Machine-Learning/Week10/Util/util.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of templates class
obj_util = foo.Util_class()


# In[3]:


# load dataset
test_dataset = pd.read_csv ("CSV_files/test_file.csv")
print("Dataset has {} rows and {} Columns".format(test_dataset.shape[0],test_dataset.shape[1])) 


# In[4]:


x_test = test_dataset.iloc[:,[2,3]].values
# label -> purchased
y_test = test_dataset.iloc[:,4].values  

print("x_test :",x_test.shape,"& y_test:",y_test.shape)


# In[5]:


#feature scalling (here data will be converted into float)
def feature_scalling(x_test,y_test):
    sc_x = StandardScaler()
    x_test = sc_x.fit_transform(x_test)
    return sc_x, x_test
    
sc_x, x_test = feature_scalling(x_test,y_test)


# In[ ]:





# In[6]:


file1 = open('LogisticRegression.pkl', 'rb')
classifier = pickle.load(file1)


# In[7]:


def final_tetsing():
    y_pre = obj_util.y_prediction(x_test, classifier)
    print("\n\n y_prediction:",y_pre)
    print(y_pre.shape)
    
    accuracy_score,average_precision,auc=obj_util.accuracy(y_pre,y_test)
    
    print('\n\nAverage accuracy_score:' , accuracy_score)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    print('Average Roc-AUC: %.3f' % auc)

    
    print("\n\nConfusion Matrix:\n",metrics.confusion_matrix(y_test, y_pre))
    obj_util.visualization(x_test,y_test, classifier, "Logistic Regression(Training set)", 
                           "Age", "Estimate Salary")
    
    obj_util.create_piklefile(classifier,'LogisticRegression.pkl' )
    print("\nPikle file created")


# In[8]:


final_tetsing()


# In[ ]:




