#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
 Support Vector Regression
1. Build a machine learning model to predict salary  based on position for a given dataset
"""


# In[63]:


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
from sklearn.preprocessing import StandardScaler
#regression librarry
from sklearn.svm import SVR
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


# In[64]:


# load dataset
dataset_original = pd.read_csv ("Position_Salaries.csv")
dataset = dataset_original
dataset.head()


# In[65]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[66]:


dataset.sample()


# In[67]:


# check dataset information
dataset.info()


# In[68]:


dataset.describe().T


# In[69]:


# handling missing data if nessesary
"""
if missing values are present
imputer = Imputer(missing_values=0, axis=0)
imputer = imputer.fit(x_data[:, 3:16])
"""
dataset.isnull().sum()


# In[70]:


# check for minimum dataset
dataset.min()


# In[72]:


#Check duplicate value
dataset.duplicated()


# In[73]:


# Divide data into features and label
x_data_set = np.array(dataset["Level"])
y_data_set = np.array(pd.DataFrame(dataset.Salary))


# In[74]:


#feature scalling (here data will be converted into float)
def feature_scalling(x_data_set,y_data_set):
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    x = sc_x.fit_transform(x_data_set.reshape(-1, 1))
    y = sc_y.fit_transform(y_data_set.reshape(-1, 1))
    
    return x, y, sc_x, sc_y
    
x, y, sc_x, sc_y = feature_scalling(x_data_set,y_data_set)


# In[75]:


print("shape of x data",x.shape)
print("shape of y data",y.shape)


# In[99]:


x


# In[76]:


# # Handle Missing data
# def handle_min_values(dataset):
#     # replace min values by mean
#     dataset.replace(0, dataset.mean(), inplace=True)
#     return dataset

# dataset = handle_min_values(dataset)


# In[11]:


# #check dataset replace with mean or not
# dataset.min()


# In[86]:


# # seperate fetures and label
# x_data = dataset.iloc[:, :-1].values
# y_data = dataset.iloc[:, 1].values


# In[62]:


# # handle categorical data
# def handle_categorical_data(x_data):
#     #encode categorical data
#     label_encod = LabelEncoder()
#     x_data[:, 1] = label_encod.fit_transform(x_data[:, 1])
    
#     # one hot encoding
#     onehotencode = OneHotEncoder(categorical_features= [1])
#     x_data = onehotencode.fit_transform(x_data).toarray()
    
#     return x_data
    
# x_data = handle_categorical_data(x_data)


# In[65]:


# #convert numpy.ndarray to DataFrame
# x_data = pd.DataFrame(x_data)
# x_data.shape


# In[37]:


# create directory to store csv files
os.mkdir("CSV_files")


# In[77]:


def csv_file(x_train_data,y_train_data,file_name):
    #load data to csv file
    myData = x_train_data
   
    myFile = open('CSV_files/'+file_name, 'w')  
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)
   
    colnames=['x'] 
    df = pd.read_csv('CSV_files/'+file_name, names=colnames, header=None)
    # inserting column with static value in data frame 
    df.insert(1, "y", y_train_data)
   
    df.to_csv('CSV_files/'+file_name, index =  False)


# In[78]:


# split dataset 
def splitdata(x, y):
    # split train and test data
    x_train,x_test,y_train,y_test= train_test_split(x, y, test_size = 1/3, random_state=0)
    print("train : ", x_train.shape,y_train.shape, " test : ", x_test.shape,y_test.shape)
    
    # saving datasets into csv files
    csv_file(x_test,y_test,'test_data.csv')

    # divide train data into train and cross validation 
    x_train_data, x_cv_data, y_train_data, y_cv_data = train_test_split(x_train,y_train,test_size = 0.40,random_state=0)
    print("train : ", x_train_data.shape,y_train_data.shape, " test : ", x_cv_data.shape,y_cv_data.shape)

    #load data into csv for train and cross validation
    csv_file(x_train_data,y_train_data,'train_data.csv')
    csv_file(x_cv_data,y_cv_data,'cv_data.csv') 

splitdata(x, y)


# In[79]:


# load dataset
train_dataset = pd.read_csv ("CSV_files/train_data.csv")
print("Dataset has {} rows and {} Columns".format(train_dataset.shape[0],train_dataset.shape[1])) 


# In[80]:


train_dataset.head()


# In[138]:


class  SupportVectorReg ():
    
    def create_module(self,x_train,y_train):
        # fitting simple LR to the training set 
        #defualt kernal for non linear module is rbf
        regressor = SVR(kernel= 'rbf')
        regressor.fit(x,y)
        return regressor

    
    def create_piklefile(self,regression):
        # dump train model pickle file
        file = open('SupportVectorReg.pkl', 'wb')
        pickle.dump(regression,file)
        file.close()          
        
    
    def y_prediction(self,x_train,regressor):
        # predicting the test set result
        # prediction for only 6.5
        y_pred_train = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
        y_pred_train = regressor.predict(x_train)
#         return y_pred_train

        print("y_predict value: ",sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]])))))
        return y_pred_train
    
    def accuracy(self,y_predict_train,y_train):
        # accuracy using r2 score
        acc_r2 = r2_score(y_train, y_predict_train)*100      
#         acc_r2 = (1-error)*100
        # Calculate accuracy using mean absolute error
        total_error = mean_absolute_error(y_train, y_predict_train)
        mean_ab = ( 1 - total_error/ len(y_train)) *100
        
        median_ab_error = median_absolute_error(y_train, y_predict_train)

        return acc_r2,mean_ab,median_ab_error

    def visualization(self,x,y,regressor):
        # Visualization the Decision Tree result (for higher resolution & smoother curve)
        print("\n visualising using SVR \n ")
        plt.scatter(x, y , color = 'pink')
        plt.plot(x, regressor.predict(x), color = 'red')
        
#         x_grid = np.arange(min(x), max(x), 0.1)
#         x_grid = x_grid.reshape((len(x_grid),1))

#         plt.scatter(x,y, color = 'pink')
#         plt.plot(x_grid, regressor.predict(x_grid), color = 'red')
        plt.title("Truth or Bulff(SVR)")
        plt.xlabel("Position Level")
        plt.ylabel("Salary")
        plt.show()
        

def main():
    #class obj created
    obj  = SupportVectorReg()
    
    # seperate fetures and label
    # here we taking only 2 columns level and salary
    x_train = train_dataset.iloc[:,:-1].values
    y_train = train_dataset.iloc[:,1].values  
    

#     print(x_train.shape, y_train.shape)
    regression = obj.create_module(x_train,y_train)
#     print("\nModule created")

    obj.create_piklefile(regression)
#     print("\nPikle file created")
    
    y_train_pre = obj.y_prediction(x_train,regression)
#     print("\n\n y_prediction:",y_train_pre)
        
    acc_r2,mean_ab,median_ab_error= obj.accuracy(y_train_pre,y_train)
    print("\n Accuracy train by acc_r2", acc_r2)
    print("\n Accuracy train by mean_ab", mean_ab)

    
    #visualisation for train dataset
    obj.visualization(x_train,y_train, regression)

if __name__ == '__main__':
    main()


# In[126]:


# Cross Validation

# load dataset
CV_dataset = pd.read_csv ("CSV_files/cv_data.csv")
print("Dataset has {} rows and {} Columns".format(CV_dataset.shape[0],CV_dataset.shape[1])) 


# In[127]:


#     #cross validation
# file1 = open('SimpleLRModulefile.pkl', 'rb')
# reg1 = pickle.load(file1)


# In[137]:


class Cross_validation():
           
    def y_prediction(self,regression, x_train):
        # predicting the test set result
        y_predict = regression.predict(x_train.reshape(-1,1))
        print("y_predict value for 6.5 is ", regression.predict(np.array(6.5).reshape(-1,1)))
        return y_predict
        
#         # predicting the test set result
#         return regression.predict(x_train)
    
    def accuracy(self,y_predict_train,y_train):
        # acc using r2
        acc_r2 = r2_score(y_train, y_predict_train)*100
#         acc_r2 = (1-error)*100
        
        # using median_ab_error
        median_ab_error = median_absolute_error(y_train, y_predict_train)
        
        total_error = mean_absolute_error(y_train, y_predict_train)
        mean_ab = ( 1 - total_error/ len(y_train)) *100
        
        return acc_r2,mean_ab,median_ab_error
    
    def visualization(self,x_test,y_test, regressor):
        # Visualization the Decision Tree result (for higher resolution & smoother curve)
        x_grid = np.arange(min(x_test), max(x_test), 0.01)
        x_grid = x_grid.reshape((len(x_grid),1))

        plt.scatter(x_test,y_test, color = 'pink')
        plt.plot(x_grid, regressor.predict(x_grid), color = 'red')
        plt.title("Truth or Bulff(SVR)")
        plt.xlabel("Position Level")
        plt.ylabel("Salary")
        plt.show()
        

def main():
    #class obj created
    obj  = Cross_validation()
    
    # seperate fetures and label
    x_cv = CV_dataset.iloc[:,:-1].values
    y_cv = CV_dataset.iloc[:,1].values
 
    #     print(x_cv.shape,y_cv.shape)
    #cross validation
    file1 = open('SupportVectorReg.pkl', 'rb')
    reg1 = pickle.load(file1)
    
    # y_prediction ( cross validation)   
    y_cv_pre = obj.y_prediction(reg1, x_cv)
    print("\n\n y_prediction:",y_cv_pre)
    
    acc_r2,mean_ab,median_ab_error= obj.accuracy(y_cv_pre,y_cv)
    print("\n Accuracy train by acc_r2", acc_r2)
    print("\n Accuracy train by mean_ab", mean_ab)
#     print("\n Accuracy train by median_ab_error", median_ab_error)

    obj.visualization(x_cv, y_cv, reg1)

if __name__ == '__main__':
    main()


# In[ ]:


# Here decision tree gives 100% or very small accuracy bcoz of overfitting and small amount of dataset


# In[ ]:




