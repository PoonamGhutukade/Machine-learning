#import the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  
import pandas as pd


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# Feature Scaling
from sklearn.preprocessing import StandardScaler
# handle categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

# Calculate Accuracy
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

import pickle 
import os, sys
import csv

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


class Util_class:
    
    # split dataset into train,test and cross validation , also load these data into csv files 
    def splitdata(self,x,y,filename, size1,size2):
        # split train and test data
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = size1, random_state=0)
        print("x_train : ", x_train.shape, " x_test : ", x_test.shape)

        # saving datasets into csv files
        #convert numpy ndarray to dataframe 
#         x_test = pd.DataFrame(x_test)
#         y_test = pd.DataFrame(y_test)
# #         data = pd.concat(x_test)
#         data = pd.concat([x_test, y_test], axis=1)
#         data.to_csv(filename,index=False,encoding='utf-8')

        #Saving testing file into pickle file
        test_file = open("CSV_files/Testing_file.csv","wb")
        pickle.dump(x_test, test_file)
        pickle.dump(y_test, test_file) 
        test_file.close()

        # divide train data into train and cross validation 
        x_train1, x_cv,  y_train1, y_cv = train_test_split(x_train,y_train, test_size = size2,random_state=0)
        print("x_train_data : ", x_train1.shape, " x_crossV_data : ", x_cv.shape)
        
        return x_train1, x_cv,  y_train1, y_cv
        
    # Feature Scaling on x_data
    def feature_Scaling(self,x_data):
        sc = StandardScaler()
        x_data = sc.fit_transform(x_data)
        return x_data
        
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
        f1_score_acc = f1_score(y_actual, y_predict)* 100  
        return accuracy_score,average_precision,auc, f1_score_acc
    
    #create confusion matrix
    def confusion_matrix(self,y_train,y_pre):
        return confusion_matrix(y_train,y_pre)
    
   
     # create pickle file dump it with module obj and file object
    def create_piklefile(self,classifier,sc, file_name):
        # dump train model pickle file
        file = open(file_name, 'wb')
        pickle.dump(classifier,file)
        pickle.dump(sc,file)
        file.close() 
        

