#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as  np
import pandas as pd
import  array
import matplotlib.pyplot as plt
import sklearn
import pickle
import warnings
import csv
warnings.filterwarnings('ignore')


# In[2]:


# load model training
file = open('training.pkl', 'rb')
regressor = pickle.load(file)


# In[3]:


# reading test csv file
data = pd.read_csv('Test/test.csv')


# In[4]:


# seperating test file into x and y
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[5]:


prediction = regressor.predict(x)


# In[6]:


# calculating accuracy 
test_accuraccy = sklearn.metrics.explained_variance_score(y,prediction)
test_accuraccy = (1-test_accuraccy)*100
print("Accuracy of test data =",  test_accuraccy)


# In[ ]:





# In[ ]:





# In[ ]:




