#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import pickle
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
import importlib.util


# In[2]:


# Saving testing file into pickle file
test_file = open("CSV_files/Testing_file.csv","rb")
X_test = pickle.load(test_file)
Y_test = pickle.load(test_file) 


# In[3]:


X_test.shape, Y_test.shape


# In[4]:


# Load xgboost model
file = open('xgboost_model.pickle', 'rb')
classifier = pickle.load(file)
file.close()


# In[5]:


# prediction
y_pred = classifier.predict(X_test)


# In[6]:


cm = confusion_matrix(Y_test, y_pred)
cm


# In[7]:


# here we devide train data into estinator number of values on rain data
accuracy = cross_val_score(estimator = classifier, X = X_test, y = Y_test, cv = 10)
print('Accuracy:', accuracy)


# In[8]:


accuracy.mean() , accuracy.std()


# In[16]:


print('The accuracy of the xgb classifier is {:.2f} out of 1 on test data'.format(classifier.score(X_test, Y_test)))


# In[ ]:




