#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering
# 1. Build a machine learning model to create group of mall customers based on their annual income and 
# spending score for a given dataset
# 

# In[2]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#libarry for feature scaling
from sklearn.preprocessing import StandardScaler

# Unsuperwised Learning-Clustering library
from sklearn.cluster import KMeans

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


# In[17]:


# seperate fetures and label
# features -> Annual Income (k$) and Spending Score (1-100)
x_train = dataset.iloc[:,[3,4]].values
print("x_train :",x_train.shape)


# In[22]:


# use elbow method to find the optimal numbers of clusters
def cluster_numbers(x_train):
    # intialize empty list for wscc(Within cluster some of Squares)
    wcss = []
    # for loop for taking multiple clusters
    for clusters in range(1, 11):
        k_means = KMeans(n_clusters= clusters, init= 'k-means++', max_iter=300, n_init= 10, random_state=0)
        k_means.fit(x_train)
        wcss.append(k_means.inertia_)
   # to display graph properly eith grids
    plt.style.use('fivethirtyeight')
    plt.plot(range(1,11),wcss)
    sb.scatterplot(range(1,11),wcss)
    plt.title("The elbow method")
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
#     print(wcss)
    
# call method 
cluster_numbers(x_train)


# In[ ]:





# # K_Means_clustering Model

# In[24]:


class K_Means_clustering():
    #apply K- means to mall dataset
    def create_module(self,x):
        k_means_obj =KMeans(n_clusters= 5, init= 'k-means++', max_iter=300, n_init= 10, random_state=0)
        return k_means_obj
    
def main():
    #class obj created
    obj  = K_Means_clustering()

    # create K_Means_clustering module
    k_means_obj = obj.create_module(x_train)
#     print("\n Module created")

    # predict y with its class
    y_kmeans = obj_util.prediction(x_train, k_means_obj)
#     print("\n y_kmean:\n",y_kmeans)
    
    # visualise customers at their class
    obj_util.visualization_for_clusters(x_train, y_kmeans, k_means_obj)
    
    # We cannot caculate accuracy here bcoz we dont have y_actual data
    
#     # Calculate accuracy  
#     adi, mib = obj_util.accuracy(y_train,y_kmeans)
    
#     print('\nPerformance by adjusted rand index:' , adi)
#     print('\nPerformance by Mutual information based score:'.mib)
    
if __name__ == '__main__':
    main()

