#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.metrics import confusion_matrix
import tensorflow as tf

import pickle


# In[6]:


# Loading testing file from pickle file
test_file = open("CSV_files/Testing_file.csv","rb")
x_test = pickle.load(test_file)
y_test = pickle.load(test_file) 

print("x_test:",x_test.shape,"y_test", y_test.shape)


# In[7]:


# load ANN model
file = open('ANN_keras.pickle', 'rb')
classifier = pickle.load(file)
sc = pickle.load(file)


# In[8]:


# Feature scaling
x_test = sc.transform(x_test) 


# In[9]:


# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)


# In[13]:


# new_prediction = classifier.predict(sc.transform(x_test))
# new_prediction = (new_prediction > 0.5)


# In[14]:


# show confusion matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)


# In[15]:


# Callculating the accuracy
print (((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% of testing data was classified correctly')


# In[ ]:





# In[ ]:




