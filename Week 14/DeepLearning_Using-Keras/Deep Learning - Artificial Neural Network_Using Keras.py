#!/usr/bin/env python
# coding: utf-8

# # Deep Learning
#     Artificial Neural Network
#     
# For a given dataset predict whether customer will exit (Output variable “Exited”) the bank or not using artificial neural network
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

# Importing the Keras libraries and packages (Using TensorFlow backend.)
import keras 
from keras.models import Sequential 
from keras.layers import Dense

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


# In[17]:


# Feature Scaling
"""
we are fitting and transforming the training data using the StandardScaler function.
We standardize our scaling so that we use the same fitted method to transform/scale test data. 
"""
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
# Data scaled properly. And done with preprocessing 
# If we fit_tranform on train data then no need to fit it again

# X_test = sc.transform(X_test)


# ### Build Artificial Neural Network Model 

# In[18]:


"""We import the required Modules here.We need the Sequential module for initializing the neural network
and the dense module to add the hidden layers."""

"""We will name the model as Classifier as our aim is to classify customer churn. 
Then we use the Sequential module for initialization."""
#Initializing Neural Network 
classifier = Sequential()


# In[19]:


x_train.shape, x_train.shape[1]


# A hurestic tip is that the amount of nodes (dimensions) in your hidden layer should be the average of your 
# input and output layers, which means that since we have 11 dimensions (representing Independent variables 
# Note: Countries still compose only one dimension) and we are looking for a binary output, we calculate 
# this to be  (11+1)÷2=6 .
# 
#     activiation: relu becasue we are in an input layer. uses the ReLu activation function for  ϕ
# 
#     nput_dim: 11 because we span 11 dimensions in our input layer. This is needed for the first added layer.
# 
#     units: 6 nodes (number of nodes in hidden layer). Can think of this as number of nodes are in the next layer.
# 
#     kernel_initializer: uniform the distribution with which we randomly initialize weights for the nodes in this layer.
#     
#     

# In[20]:


# 1st layer (addthe input layer and the first hidden layer −)
# Use dense module to add the hidden layers.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

# 2nd layer (bcoz we want to implement Deep Learning,which is an ANN with many layers)
# we do not need to specify input dim. 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# output layer
# sigmoid used instead of the ReLu function becasue it generates probabilities for the outcome. 
# We want the probability that each customer leaves the bank.
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# ###### If we want more than two categories, then we will need to change
# 1) the units parameter to match the desired category count
# 
# 2) the activation field to softmax. Basically a sigmoid function but applied to a dependent variable that has more than 2 categories.

# #### Compiling the Neural network
#     
#     optimizer: adam The algorithm we want to use to find the optimal set of weights in the neural networks. 
# Adam is a very efficeint variation of Stochastic Gradient Descent.
#     
#     loss: binary_crossentropy This is the loss function used within adam. This should be the logarthmic loss.
# If our dependent (output variable) is Binary, it is binary_crossentropy. If Categorical, then it is called categorical_crossentropy

# In[21]:


# Compiling Neural Network 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# #### Fitting the Neural Network
# 
#     batch_size: How often we want to back-propogate the error values so that individual node weights can be adjusted.
# 
#     nb_epochs: The number of times we want to run the entire test data over again to tune the weights. This is like the fuel of the algorithm.

# In[22]:


# Fitting ANN to the training set
classifier.fit(x_train, y_train, batch_size = 15, epochs = 60)


# ### Save ANN model 

# In[31]:


file = open('ANN_keras.pickle', 'wb')
pickle.dump(classifier, file)
pickle.dump(sc,file)
file.close()


# #### Testing the ANN
# 

# In[32]:


# Predicting the Test set results
y_pred = classifier.predict(x_cv)
y_pred = (y_pred > 0.5)


# In[33]:


y_pred


# In[34]:


# new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 500, 1, 40, 3, 50000, 2, 1, 1, 40000]])))
new_prediction = classifier.predict(sc.transform(x_cv))
new_prediction = (new_prediction > 0.5)


# Significance of the confusion matrix value:
# The output should be close to the table below:
# 
#         Predicted: No  Predicted: Yes
#     Actual:No 1504             91
#     Actual:Yes 184           221
#     
# This means that we should have about  (1504+221)=1726  
# correct classifications out of our total testing data size of  2000 . 
# This means that our accuracy for this trial was  1726÷2000=86.3% , which matches the classifier's prediction

# In[35]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_cv, new_prediction)
print (cm)


# In[36]:


# Callculating the accuracy
print (((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% of testing data was classified correctly')


# In[37]:


Accuracy =((1288+0)/1600)*100
print(Accuracy)


# In[38]:


1288+312


# In[ ]:





# In[ ]:




