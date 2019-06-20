#!/usr/bin/env python
# coding: utf-8

# # XGBOOST
#     
#     For a given dataset predict whether customer will exit (Output variable “Exited”) the bank or not using artificial neural network
# 

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
import pickle
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
import importlib.util


# In[2]:


# Importing the database
dataset = pd.read_csv('Churn_Modelling.csv')


# ### Data preprocessing

# In[3]:


dataset.head()


# In[4]:


# look dataset information
dataset.info()


# In[5]:


# Check for minimum and maximum values for column in dataset
dataset.describe().T


# In[6]:


# Checking for null values
dataset.isnull().sum()


# In[7]:


# Checking for duplicate values
dataset.duplicated().sum()


# In[8]:


# boxplot to  find outliers of datatset
sb.boxplot(data = dataset)


# In[9]:


#chekc correlation of target "y" with each other data
corr = dataset.corr()
sb.heatmap(corr)


# In[10]:


# easy way to check correlation of target variable to all oather features after getdummies
print (corr['Exited'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['Exited'].sort_values(ascending=False)[-10:]) #last 5 values`


# In[11]:


# We create matrices of the features of dataset and the target variable,
# split datasetb into x(features) and y(label
x_data = dataset.iloc[:, 3:13].values
y_data = dataset.iloc[:, 13].values
x_data


# In[12]:


y_data


# In[13]:


# We will do the same thing for gender. this will be binary in this dataset
print(x_data[:6,2], '... will now become: ')


# #### Handling categorical data

# In[14]:


"""
#Geography and Gender has a object datatype(string variables) we have to encode it
We make the analysis simpler by encoding string variables.
We are using the ScikitLearn function ‘LabelEncoder’ to automatically encode the different labels
in the columns with values between 0 to n_classes-1.
"""

labelencoder_X_1 = LabelEncoder() 
x_data[:,1] = labelencoder_X_1.fit_transform(x_data[:,1])

labelencoder_X_2 = LabelEncoder() 
x_data[:, 2] = labelencoder_X_2.fit_transform(x_data[:, 2])
x_data


# In[15]:


"""We use the same ScikitLearn library and another function called the OneHotEncoder to just pass the column
number creating a dummy variable."""
onehotencoder = OneHotEncoder(categorical_features = [1])
x_data = onehotencoder.fit_transform(x_data).toarray()
x_data = x_data[:, 1:]
x_data


# In[16]:


# import os
# os.mkdir('CSV_files')


# In[17]:


#Splitting the dataset into the Training set and the Test Set
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = 0.2)
print("x_train : ", X_train.shape, " x_test : ", X_test.shape)

# Saving testing file into pickle file
test_file = open("CSV_files/Testing_file.csv","wb")
pickle.dump(X_test, test_file)
pickle.dump(Y_test, test_file) 
test_file.close()

# split dataset into training and crossvalidation set
x_train, x_cv, y_train, y_cv = train_test_split(X_train, Y_train, test_size = 0.20)
print("x_train_data : ", x_train.shape, " x_crossV_data : ", x_cv.shape)


# In[18]:


from xgboost import XGBClassifier


# In[19]:


# Fitting the xgboost to the training set
classifier = XGBClassifier()
classifier.fit(x_train, y_train)


# In[20]:


# prediction
y_pred = classifier.predict(x_cv)


# In[22]:


cm = confusion_matrix(y_cv, y_pred)
cm


# In[23]:


from sklearn.model_selection import cross_val_score


# In[24]:


# here we devide train data into estinator number of values on rain data
accuracy = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print('Accuracy:', accuracy)


# In[25]:


accuracy.mean() , accuracy.std()


# In[26]:


print('The accuracy of the xgb classifier is {:.2f} out of 1 on training data'.format(classifier.score(x_train, y_train)))
print('The accuracy of the xgb classifier is {:.2f} out of 1 on test data'.format(classifier.score(x_cv, y_cv)))


# #### Store model into pickle file

# In[27]:


file = open('xgboost_model.pickle', 'wb')
pickle.dump(classifier, file)
file.close()


# ## SVM 

# In[28]:


from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(x_train, y_train)

print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(x_train, y_train)))

print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(x_cv, y_cv)))


# In[40]:


# here we devide train data into estinator number of values on rain data
accuracy = cross_val_score(estimator = svm, X = x_train, y = y_train, cv = 10)
print('Accuracy:', accuracy)


# In[41]:


accuracy.mean() , accuracy.std()


# ## KNN

# In[43]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(x_train, y_train)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(x_train, y_train)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(x_cv, y_cv)))


# In[44]:


# here we devide train data into estinator number of values on rain data
accuracy = cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 10)
print('Accuracy:', accuracy)


# In[45]:


accuracy.mean() , accuracy.std()


# In[ ]:




