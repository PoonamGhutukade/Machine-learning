#import the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  
import pandas as pd
#imputer to handle missing data 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
# handle categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#For feature scaling
from sklearn.preprocessing import StandardScaler

#confusion matix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#visualisation
from matplotlib.colors import ListedColormap
# for encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing 
from collections import defaultdict
#o check accuracy
from sklearn.metrics import accuracy_score
# calculate accuracy
from sklearn import metrics
# to check accuracy
from sklearn.metrics import *

import pickle 
import os, sys
import csv

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


class Util_class:
    
    # split dataset into train,test and cross validation , also load these data into csv files 
    def splitdata(self,dataset,size1, size2, dir_name):
        # split train and test data
        train, test = train_test_split(dataset,test_size = size1, random_state=0)
        print("train : ", train.shape, " test : ", test.shape)

        # saving datasets into csv files
        test.to_csv(dir_name+'/test_file.csv',index=False,encoding='utf-8')

        # divide train data into train and cross validation 
        train_data, crossV_data = train_test_split(train,test_size = size2,random_state=0)
        print("train_data : ", train_data.shape, " crossV_data : ", crossV_data.shape)
        
        return train_data, crossV_data
        
    # Handle categorical data
    def Categorical_data(self,df_new):
        # from collections import defaultdict
        d = defaultdict(LabelEncoder)
        # Encoding the variable
        fit = df_new.apply(lambda x: d[x.name].fit_transform(x))
        # Inverse the encoded
        fit.apply(lambda x: d[x.name].inverse_transform(x))
        # Using the dictionary to label future data
        df_new.apply(lambda x: d[x.name].transform(x))
        # one hot encoding
        obj_oneh = OneHotEncoder()
        obj_oneh.fit(df_new)
        new_dataset = obj_oneh.transform(df_new).toarray()
        return new_dataset
    
    # predict y on given y data
    def prediction(self, x, model_obj):
        # fit predict method, put client who is belongs to which cluster exactly
        y_pred = model_obj.fit_predict(x)
        return y_pred
    
    # calculate accuracy
    def accuracy(self,y_actual, y_predict): 
        # performance adjusted rand index
        adi = metrics.adjusted_rand_score(y_actual, y_predict) 
        # Mutual information based score
        mib =  metrics.adjusted_mutual_info_score(y_actual, y_predict)        
        return adi, mib
    
    # cluster visualisation for K-means
    def visualization_for_clusters(self,x, y_kmeans, k_means_obj):
        plt.figure(figsize = (16, 10))
        # to display graph properly eith grids
        plt.style.use('fivethirtyeight')
        # all classes
        plt.scatter(x[y_kmeans == 0 , 0], x[y_kmeans == 0 ,1], s = 100, c = 'red', label = 'Careful')
        plt.scatter(x[y_kmeans == 1 , 0], x[y_kmeans == 1 ,1], s = 100, c = 'blue', label = 'Standard')
        plt.scatter(x[y_kmeans == 2 , 0], x[y_kmeans == 2 ,1], s = 100, c = 'cyan', label = 'Target')
        plt.scatter(x[y_kmeans == 3 , 0], x[y_kmeans == 3 ,1], s = 100, c = 'green', label = 'Careless')
        plt.scatter(x[y_kmeans == 4 , 0], x[y_kmeans == 4 ,1], s = 100, c = 'magenta', label = 'Sensible')
        
        # center of all class
        plt.scatter(k_means_obj.cluster_centers_[:, 0],k_means_obj.cluster_centers_[:, 1],marker='*', s = 300, c = 'black', label= 'centroid')
        plt.title('cluster of client', fontsize = 20)
        plt.xlabel('Annual income')
        plt.ylabel('Spending score')
        plt.legend()
        plt.show()
        
       
    # cluster visualisation for Hierarchical clusterin
    def visualization_for_clusters_hc(self,x, y_hcm, hcm):
        plt.figure(figsize = (16, 10))
        # to display graph properly eith grids
        plt.style.use('fivethirtyeight')
        # all classes
        plt.scatter(x[y_hcm == 0 , 0], x[y_hcm == 0 ,1], s = 100, c = 'olive', label = 'Careful')
        plt.scatter(x[y_hcm == 1 , 0], x[y_hcm == 1 ,1], s = 100, c = 'darkblue', label = 'Standard')
        plt.scatter(x[y_hcm == 2 , 0], x[y_hcm == 2 ,1], s = 100, c = 'rosybrown', label = 'Target')
        plt.scatter(x[y_hcm== 3 , 0], x[y_hcm == 3 ,1], s = 100, c = 'steelblue', label = 'Careless')
        plt.scatter(x[y_hcm== 4 , 0], x[y_hcm== 4 ,1], s = 100, c = 'hotpink', label = 'Sensible')
        plt.title('cluster of client')
        plt.xlabel('Annual income')
        plt.ylabel('Spending score')
        plt.legend()
        plt.show()
    
   
     # create pickle file dump it with module obj and file object
    def create_piklefile(self,classifier, file_name):
        # dump train model pickle file
        file = open(file_name, 'wb')
        pickle.dump(classifier,file)
        file.close() 
        

