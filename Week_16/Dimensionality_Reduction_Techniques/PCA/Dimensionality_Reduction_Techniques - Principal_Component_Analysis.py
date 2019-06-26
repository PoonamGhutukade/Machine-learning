#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction Techniques
# 		Principal Component Analysis
# 1.Apply PCA for a dataset given in the url, build a classification model and plot the graph
# 

# In[1]:


# import all the libraries
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import pickle


# In[2]:


# load the dataset
dataset = pd.read_csv('Wine.csv') 
dataset.head()


# In[3]:


dataset.info()


# In[4]:


dataset.describe().T


# In[5]:


# Checking for the null values
dataset.isnull().sum()


# In[6]:


# checking for the duplicate values
dataset.duplicated().sum()


# In[7]:


# devide the data into x(independent variables 0 to 12) and y(dependent variable 13th) 
x_data = dataset.iloc[:, 0:13].values
y_data = dataset.iloc[:, 13].values

x_data.shape, y_data.shape


# ### Preprocessing

# In[8]:


# import os,sys
# os.mkdir('CSV_files')


# In[9]:


# split dataset into train,test and cross validation , also load these data into csv files 
def splitdata(x,y,size1,size2):
    # split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = size1, random_state=0)
    print("x_train : ", x_train.shape, " x_test : ", x_test.shape)

    #Saving testing file into pickle file
    test_file = open("CSV_files/Testing_file.csv","wb")
    pickle.dump(x_test, test_file)
    pickle.dump(y_test, test_file) 
    test_file.close()

    # divide train data into train and cross validation 
    x_train1, x_cv,  y_train1, y_cv = train_test_split(x_train,y_train, test_size = size2,random_state=0)
    print("x_train_data : ", x_train1.shape, " x_crossV_data : ", x_cv.shape)

    return x_train1, x_cv,  y_train1, y_cv
        
x_train, x_cv,  y_train, y_cv = splitdata(x_data,y_data,0.2,0.2)


# In[10]:


# Feature Scaling on x_data
def feature_Scaling(x_data, x_cv):
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)
    x_cv = sc.transform(x_cv)
    return sc, x_data, x_cv
sc, x_train, x_cv = feature_Scaling(x_train, x_cv)


# In[11]:


#Applying PCA
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_cv = pca.transform(x_cv)
# explained_varience = pca.explained_variance_ratio_


# In[12]:


x_train.shape, x_cv.shape


# In[13]:


# When PCA(n_components = 2)
print('X_train:',x_train.shape), print('X_cv',x_cv.shape)
x_train


# ### Load train model into Logistic regression

# In[14]:


# fitting logistinc regression to the training set
classifier = LogisticRegression(random_state=0)
classifier = classifier.fit(x_train,y_train)


# In[15]:


# predict y data
y_pred = classifier.predict(x_cv)


# In[16]:


cm = confusion_matrix(y_cv, y_pred)
print(cm)


# ### Accuracy 

# In[17]:


print(classification_report(y_cv, y_pred))


# ### Visualisation

# In[18]:


X_set, Y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('cyan', 'yellow', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
               c = ListedColormap(('red', 'green', 'white'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# ### Store train model into pickle file

# In[19]:


file1 = open('Train_model.pickle','wb')
pickle.dump(classifier, file1)
pickle.dump(sc, file1)
# pickle.dump(pca, file1)
file1.close()


# In[20]:


file2 = open('Train_model1.pickle','wb')
# pickle.dump(classifier, file2)
# pickle.dump(sc, file2)
pickle.dump(pca, file2)
file2.close()


# In[ ]:




