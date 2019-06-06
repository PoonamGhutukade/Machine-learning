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

#Classification library
from sklearn.linear_model import LogisticRegression
#confusion matix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#visualisation
from matplotlib.colors import ListedColormap
# for encoding
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

         #load data into csv for train and cross validation
        train_data.to_csv(dir_name+'/train_file.csv',index=False,encoding='utf-8')
        crossV_data.to_csv(dir_name+'/CValidation_file.csv',index=False,encoding='utf-8')

        print("train_data : ", train_data.shape, " crossV_data : ", crossV_data.shape)
        
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
    def y_prediction(self,x_data,classifier):
        # predicting the train set result
        y_predict = classifier.predict(x_data)
        return y_predict

    
    def accuracy(self,y_predict,y_actual):
        # calculate accuracy 
        accuracy_score = metrics.accuracy_score(y_actual, y_predict) * 100
        average_precision = average_precision_score(y_actual, y_predict) * 100
        auc = roc_auc_score(y_actual, y_predict)* 100   
        return accuracy_score,average_precision,auc
    
    def visualization(self,x_train,y_train,classifier,title, xlabel, ylabel ):
        # Visualization the Decision Tree result (for higher resolution & smoother curve) 
        x_set, y_set = x_train, y_train 
        x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), 
                     alpha = 0.75, cmap = ListedColormap(('red' , 'orange')))

        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
#         print(x_set[:10],'\n',y_set[:10],'\n')
#         print(x_set.shape)
        for i,j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j,0], x_set[y_set == j,1],c = ListedColormap(("cyan", "blue"))(i),label = j)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    
    #create confusion matrix
    def confusion_matrix(self,y_train,y_pre):
        return confusion_matrix(y_train,y_pre)
    
   
     # create pickle file dump it with module obj and file object
    def create_piklefile(self,classifier, file_name):
        # dump train model pickle file
        file = open(file_name, 'wb')
        pickle.dump(classifier,file)
        file.close() 
        

