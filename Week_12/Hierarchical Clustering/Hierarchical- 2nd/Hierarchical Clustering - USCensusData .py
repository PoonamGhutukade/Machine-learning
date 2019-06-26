#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Clustering
# 2. Apply K-Means clustering on below dataset

# In[27]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#libarry for feature scaling
from sklearn.preprocessing import StandardScaler

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


# In[2]:


# importing template file 
spec = importlib.util.spec_from_file_location("Util_class", "/home/admin1/PycharmProjects/Machine-Learning/Week 12/Util/util.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
obj_util = foo.Util_class()


# In[3]:


# read the large csv file with specified chunksize 
df = pd.read_csv ("USCensus1990.data.txt",delimiter=",", sep='\t', iterator=True, chunksize=10000)


# In[4]:


# convert textfile format to pandas dataframe
dataset = pd.concat(df, ignore_index=True)


# In[5]:


dataset.head()


# In[6]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[7]:


# check dataset information
dataset.info()


# In[8]:


# descibe the dataset
# dataset.describe().T


# In[9]:


# handling missing data if nessesary
"""
if missing values are present
imputer = Imputer(missing_values=0, axis=0)
imputer = imputer.fit(x_data[:, 3:16])
"""
dataset.isnull().sum()


# In[10]:


# check for duplicate values
dataset.duplicated().sum()


# In[11]:


#Display heatmap to show correlation between diff variables
corr = dataset.corr()
corr


# In[12]:


sb.heatmap(corr)


# In[ ]:





# In[19]:


# x_data = dataset.iloc[:,[3,4]].values
x_data = dataset[:10000]
print("x_data :",x_data.shape)


# In[22]:


# Using the dendrogram to find the optimal number of clusters
def show_dendrogram(x_data):
    plt.figure(figsize = (16, 10))
    # ward method => Perform Ward’s linkage on a condensed distance matrix.
    dendrogram = sch.dendrogram(sch.linkage(x_data, method = 'ward'))
    plt.title("Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel('Euclidean Distance')
    plt.show()
show_dendrogram(x_data)


# In[21]:





# # Hierarchical_clustering Model

# In[34]:


class Hierarchical_clustering():
    
    #fitting Hierarchical_clustering to the mall dataset
    def create_module(self,x):
        hcm = AgglomerativeClustering(n_clusters = 3, affinity='euclidean', linkage= 'ward')
        return hcm
    
    # cluster visualisation for Hierarchical clusterin
    def visualization_for_clusters_hc(self,x, y_kmeans, k_means_obj):
#         plt.figure(figsize = (16, 10))
        # to display graph properly eith grids
        plt.style.use('fivethirtyeight')
        # all classes
        plt.scatter(x[y_kmeans == 0 , 0], x[y_kmeans == 0 ,1], s = 100, c = 'olive', label = 'Cluster1')
        plt.scatter(x[y_kmeans == 1 , 0], x[y_kmeans == 1 ,1], s = 100, c = 'darkblue', label = 'Cluster2')
        plt.scatter(x[y_kmeans == 2 , 0], x[y_kmeans == 2 ,1], s = 100, c = 'rosybrown', label = 'Cluster3')
#         plt.scatter(x[y_kmeans == 3 , 0], x[y_kmeans == 3 ,1], s = 100, c = 'steelblue', label = 'Careless')
#         plt.scatter(x[y_kmeans == 4 , 0], x[y_kmeans == 4 ,1], s = 100, c = 'hotpink', label = 'Sensible')
        plt.title('cluster of client')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend()
        plt.show()
    
def main():
    #class obj created
    obj  = Hierarchical_clustering()

    # create  Hierarchical_clustering module
    hcm = obj.create_module(x_data)
    print("\n Module created")

    # predict y with its class
    y_hcm = obj_util.prediction(x_data, hcm)
#     print("\n y_hcm:\n",y_hcm)
    
    # visualise customers at their class
#     obj.visualization_for_clusters_hc(x_data, y_hcm, hcm)
    
    # We cannot caculate accuracy here bcoz we dont have y_actual data
    
#     # Calculate accuracy  
#     adi, mib = obj_util.accuracy(y_train,y_kmeans)
    
#     print('\nPerformance by adjusted rand index:' , adi)
#     print('\nPerformance by Mutual information based score:'.mib)
    

if __name__ == '__main__':
    main()


# In[ ]:




