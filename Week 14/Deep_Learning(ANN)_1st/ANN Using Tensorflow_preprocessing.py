#!/usr/bin/env python
# coding: utf-8

# # Deep Learning
#     Artificial Neural Network
#          For a given dataset predict whether customer will exit (Output variable “Exited”) the bank or not       using artificial neural network
# 

# We will follow the following steps
# 
#     1. Importing the libraries & Dataset
#     2. Encoding Categorical data
#     3. Splitting the Dataset into Training, Cross Validation & Test set, Feature Scaling
#     4. Creating & Compiling ANN (Artificial Neural Network)
#     5. Applying ANN to the training dataset.
#     6. Predicting the outcome using ANN.

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import tensorflow as tf

import pickle
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
import importlib.util


# In[2]:


# Importing the database
dataset = pd.read_csv('Churn_Modelling.csv')


# ### Dataset Preprocessing

# In[7]:


dataset.head()


# In[8]:


# look dataset information
dataset.info()


# In[9]:


# Check for minimum and maximum values for column in dataset
dataset.describe().T


# In[10]:


# Checking for null values
dataset.isnull().sum()


# In[11]:


# Checking for duplicate values
dataset.duplicated().sum()


# In[12]:


"""Handling Missing Data"""
def missing_data(df):
        
        #check for minimum values
        print("\n\nMin values:\n",  df.min())
        
        df.replace(np.NaN, df.mode, inplace = True)
        
        print("\n After replacing minimum values\n",df.min())
        
        # check for duplicate data
#         df.duplicated().sum()
#         print("\nCheck duplicate values:\n",df.duplicated().sum())
missing_data(dataset)


# In[13]:


#chekc correlation of target "y" with each other data
corr = dataset.corr()
sb.heatmap(corr)


# In[14]:


# easy way to check correlation of target variable to all oather features after getdummies
print (corr['Exited'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['Exited'].sort_values(ascending=False)[-10:]) #last 5 values`


# In[15]:


# df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)


# In[16]:


# We create matrices of the features of dataset and the target variable,
# split datasetb into x(features) and y(label
x_data = dataset.iloc[:, 3:13].values
y_data = dataset.iloc[:, 13].values
print("x_data: ",x_data.shape, "y_data: ",y_data.shape)
x_data


# In[17]:


y_data


# ### Encoding the categirical data using LabelEncoder & OneHotEncoder

# In[18]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
"""
#Geography and Gender has a object datatype(string variables) we have to encode it
We make the analysis simpler by encoding string(OBJECT type) variables.
We are using the ScikitLearn function ‘LabelEncoder’ to automatically encode the different labels
in the columns with values between 0 to n_classes-1.
"""
labelencoder_X_1 = LabelEncoder() 
x_data[:,1] = labelencoder_X_1.fit_transform(x_data[:,1])

labelencoder_X_2 = LabelEncoder() 
x_data[:, 2] = labelencoder_X_2.fit_transform(x_data[:, 2])
"""
It label France=0,  Spain=2,  Germany= 1 using LabelEncoder
"""
print(x_data.shape)
x_data


# In[19]:


"""We use the same ScikitLearn library and another function called the OneHotEncoder to just pass the column
number creating a dummy variable."""
onehotencoder = OneHotEncoder(categorical_features = [1])
x_data = onehotencoder.fit_transform(x_data).toarray()
x_data = x_data[:, 1:]
print(x_data.shape)
x_data


# In[21]:


type(y_data)


# ### Splitting the dataset into train, test & cross_validation

# In[28]:


#Splitting the dataset into the Training set and the Test Set
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = 0.2)
print("x_train : ", X_train.shape, " x_test : ", X_test.shape)

# Saving testing file into pickle file
test_file = open("CSV_files/Testing_file.pickle","wb")
pickle.dump(X_test, test_file)
pickle.dump(Y_test, test_file) 
test_file.close()

# split dataset into training and crossvalidation set
x_train, x_cv, y_train, y_cv = train_test_split(X_train, Y_train, test_size = 0.20)
print("x_train_data : ", x_train.shape, " x_crossV_data : ", x_cv.shape)


# #### Save Preprocessing file

# In[29]:


# Save preprocessing file 
with open('CSV_files/Preprocessing_file.pickle','wb') as f:
        pickle.dump([x_train,y_train,x_cv,y_cv],f)


# In[1]:


#####################################------END------#################################


# In[ ]:




