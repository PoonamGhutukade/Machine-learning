#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  
import pandas as pd
#imputer to handle missing data 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
# handle categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#regression librarry
from sklearn.ensemble import RandomForestRegressor
#o check accuracy
from sklearn.metrics import accuracy_score
# to check accuracy
from sklearn.metrics import *

import pickle 
import os, sys
import csv

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


class Util_class:
    
#     def print_data(para):
#         print("Hello!!!",para)
#         return
    
    # split dataset into train,test and cross validation , also load these data into csv files 
    def splitdata(dataset,size1, size2):
        # split train and test data
        train, test = train_test_split(dataset,test_size = size1, random_state=0)
        print("train : ", train.shape, " test : ", test.shape)

        # saving datasets into csv files
        test.to_csv('CSV_files/test_file.csv',index=False,encoding='utf-8')

        # divide train data into train and cross validation 
        train_data, crossV_data = train_test_split(train,test_size = size2,random_state=0)

         #load data into csv for train and cross validation
        train_data.to_csv('CSV_files/train_file.csv',index=False,encoding='utf-8')
        crossV_data.to_csv('CSV_files/CValidation_file.csv',index=False,encoding='utf-8')

        print("train_data : ", train_data.shape, " crossV_data : ", crossV_data.shape)
        
    #create module for classification for RandomForestRegressor
    def create_module(x_train,y_train):
        # fitting simple LR to the training set
        regression = RandomForestRegressor(n_estimators= 300 ,random_state=0)
        regression.fit(x_train,y_train)
        return regression
    
    # create pickle file dump it with module obj and file object
    def create_piklefile(regression, file_name):
        # dump train model pickle file
        file = open(file_name, 'wb')
        pickle.dump(regression,file)
        file.close() 
        
    # predict y on given y data
    def y_prediction(regression, x_train):
        print("regression object",type(regression))
        # predicting the test set result
        y_predict = regression.predict(x_train)
        print("y_predict value for 6.5 is ", regression.predict(np.array(6.5).reshape(-1,1)))
        return y_predict
        
#         # predicting the test set result
#         return regression.predict(x_train)
    
    def accuracy(y_predict_train,y_train):
        # accuracy using r2 score
        acc_r2 = r2_score(y_train, y_predict_train)*100      
#         acc_r2 = (1-error)*100
  
        total_error = mean_absolute_error(y_train, y_predict_train)
        mean_ab=( 1- (total_error / len(y_train))) *100
        
        mean_sq  = mean_squared_error(y_train, y_predict_train) 

        mean_sq_log = mean_squared_log_error(y_train, y_predict_train)  
    
        median_ab_error = median_absolute_error(y_train, y_predict_train)
        
        return acc_r2,mean_ab,mean_sq,mean_sq_log, median_ab_error
    
    def visualization(x_test,y_test, regression, title, xlabel, ylabel):
        # Visualization the Decision Tree result (for higher resolution & smoother curve)
        x_grid = np.arange(min(x_test), max(x_test), 0.01)
        x_grid = x_grid.reshape((len(x_grid),1))
        
        plt.scatter(x_test,y_test, color = 'red')
#         plt.plot(x_grid,regression.predict(x_grid), color = 'blue')
        # reshape x_grid or not both will give same ploting
        plt.plot(x_grid,regression.predict(x_grid.reshape(-1,1)), color = 'blue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()    
    

