#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Clustering	
# 1. Build a machine learning model to create group of mall customers based on their annual income and
# spending score for a given dataset
# 

# In[2]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Unsuperwised Learning-Clustering library
from sklearn.cluster import KMeans
# Their are two types of clutering - 1) AgglomerativeClustering 2) DIvisive
from sklearn.cluster import AgglomerativeClustering
# to draw dendrogram
import scipy.cluster.hierarchy as sch

import pickle 
import os, sys
import csv

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import importlib.util


# In[3]:


# importing template file 
spec = importlib.util.spec_from_file_location("Util_class", "/home/admin1/PycharmProjects/Machine-Learning/Week 12/Util/util.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
obj_util = foo.Util_class()


# In[4]:


# load dataset
dataset_original = pd.read_csv ("Mall_Customers.csv")
dataset = dataset_original
dataset.head()


# In[5]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[6]:


# check dataset information
dataset.info()


# In[7]:


# descibe the dataset
dataset.describe().T


# In[8]:


# handling missing data if nessesary
"""
if missing values are present
imputer = Imputer(missing_values=0, axis=0)
imputer = imputer.fit(x_data[:, 3:16])
"""
dataset.isnull().sum()


# In[9]:


# check for duplicate values
dataset.duplicated().sum()


# In[10]:


#Display heatmap to show correlation between diff variables
corr = dataset.corr()
sb.heatmap(corr)


# In[11]:


# # check skewness for target variable
# sb.distplot(dataset['Spending Score (1-100)'])
# print ("Skewness of y is {}".format(dataset['Spending Score (1-100)'].skew()))


# In[12]:


# print (corr['Spending Score (1-100)'].sort_values(ascending=False)[:10], '\n') #top 10 values
# print ('----------------------')
# print (corr['Spending Score (1-100)'].sort_values(ascending=False)[-5:]) #last 5 values`


# In[17]:


# seperate fetures and label

# features -> Annual Income (k$) and Spending Score (1-100)
x_train = dataset.iloc[:,[3,4]].values
print("x_train :",x_train.shape)


# ## Dendrogram

# """
# Dendrogram():
# The hierarchy class has a dendrogram method which takes the value returned by the linkage method of the same class.
# The linkage method takes the dataset and the method to minimize distances as parameters.
# We use 'ward' as the method since it minimizes then variants of distances between the clusters."""

# In[18]:


# Using the dendrogram to find the optimal number of clusters
def show_dendrogram(x_train):
    plt.figure(figsize = (16, 10))
    # ward method => Perform Ward’s linkage on a condensed distance matrix.
    dendrogram = sch.dendrogram(sch.linkage(x_train, method = 'ward'))
    plt.title("Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel('Euclidean Distance')
    plt.show()
show_dendrogram(x_train)


# # Hierarchical_clustering Model

# In[20]:


class Hierarchical_clustering():
    
    #fitting Hierarchical_clustering to the mall dataset
    def create_module(self,x):
        hcm = AgglomerativeClustering(n_clusters= 5, affinity='euclidean', linkage= 'ward')
        return hcm
    
def main():
    #class obj created
    obj  = Hierarchical_clustering()

    # create  Hierarchical_clustering module
    hcm = obj.create_module(x_train)
#     print("\n Module created")

    # predict y with its class
    y_hcm = obj_util.prediction(x_train, hcm)
#     print("\n y_hcm:\n",y_hcm)
    
    # visualise customers at their class
    obj_util.visualization_for_clusters_hc(x_train, y_hcm, hcm)
    
    # We cannot caculate accuracy here bcoz we dont have y_actual data
    
#     # Calculate accuracy  
#     adi, mib = obj_util.accuracy(y_train,y_kmeans)
    
#     print('\nPerformance by adjusted rand index:' , adi)
#     print('\nPerformance by Mutual information based score:'.mib)
    

if __name__ == '__main__':
    main()

