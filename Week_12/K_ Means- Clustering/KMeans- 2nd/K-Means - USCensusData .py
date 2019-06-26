#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering
# 2. Apply K-Means clustering on below dataset

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


# read the large csv file with specified chunksize 
df = pd.read_csv ("USCensus1990.data.txt",delimiter=",", sep='\t', iterator=True, chunksize=10000)


# In[5]:


# convert textfile format to pandas dataframe
dataset = pd.concat(df, ignore_index=True)


# In[6]:


dataset.head()


# In[7]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[8]:


# check dataset information
dataset.info()


# In[10]:


# descibe the dataset
dataset.describe()


# In[11]:


# handling missing data if nessesary
"""
if missing values are present
imputer = Imputer(missing_values=0, axis=0)
imputer = imputer.fit(x_data[:, 3:16])
"""
dataset.isnull().sum()


# In[12]:


# check for duplicate values
dataset.duplicated().sum()


# In[16]:


#Display heatmap to show correlation between diff variables
corr = dataset.corr()
corr


# In[17]:


sb.heatmap(corr)


# In[ ]:





# In[10]:


# x_data = dataset.iloc[:,[3,4]].values
x_data = dataset[:100000]
print("x_data :",x_data.shape)


# In[ ]:





# In[17]:


# use elbow method to find the optimal numbers of clusters
def cluster_numbers(x_data):
    # intialize empty list for wscc(Within cluster some of Squares)
    wcss = []
    # for loop for taking multiple clusters
    for clusters in range(1, 11):
        k_means = KMeans(n_clusters= clusters, init= 'k-means++', max_iter=300, n_init= 10, random_state=0)
        k_means.fit(x_data)
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
cluster_numbers(x_data)


# In[ ]:





# # K_Means_clustering Model

# In[45]:


class K_Means_clustering():
    #apply K- means to mall dataset
    def create_module(self,x):
        k_means_obj =KMeans(n_clusters = 3, init= 'k-means++', max_iter=30, n_init= 10, random_state=0)
        return k_means_obj
                                            
    def visualization_for_clusters(self,x_data, y_kmeans, k_means_obj):
        colors=['orange', 'blue', 'green']
        data = x_data.values[:, 0:69]
        category = x_data.values[:, 68]
                        
        for i in range(3):
#             plt.scatter(x[y_kmeans == 0 , 0], x[y_kmeans == 0 ,1], s = 100, c = 'red', label = 'Careful')
            plt.scatter(x_data[y_kmeans == i, 0], x_data[y_kmeans == i,1], marker='D', s=100, color = colors[int(category[i])])
        plt.scatter(k_means_obj.cluster_centers_[:,0], k_means_obj.cluster_centers_[:,1], marker='*', c='g', s=150)
    
def main():
    #class obj created
    obj  = K_Means_clustering()

    # create K_Means_clustering module
    k_means_obj = obj.create_module(x_data)
    print("\n Module created")

    # predict y with its class
    y_kmeans = obj_util.prediction(x_data, k_means_obj)
#     print("\n y_kmean:\n",y_kmeans)
    
    # visualise customers at their class
#     obj.visualization_for_clusters(x_data, y_kmeans, k_means_obj)
#     obj_util.visualization_for_clusters(x_data, y_kmeans, k_means_obj)
    # Plot the data and the centers generated as random

    # We cannot caculate accuracy here bcoz we dont have y_actual data
    
#     # Calculate accuracy  
#     adi, mib = obj_util.accuracy(y_train,y_kmeans)
    
#     print('\nPerformance by adjusted rand index:' , adi)
#     print('\nPerformance by Mutual information based score:'.mib)
    
if __name__ == '__main__':
    main()


# In[ ]:




