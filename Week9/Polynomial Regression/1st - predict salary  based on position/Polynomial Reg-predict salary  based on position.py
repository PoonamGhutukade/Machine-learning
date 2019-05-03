#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Polynomial Regression
1. Build a machine learning model to predict salary  based on position for a given dataset

"""


# In[2]:


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
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures

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


# In[3]:


# load dataset
dataset_original = pd.read_csv ("Position_Salaries.csv")
dataset = dataset_original
dataset.head()


# In[4]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[5]:


dataset.sample()


# In[6]:


# check dataset information
dataset.info()


# In[7]:


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


# check for minimum dataset
dataset.min()


# In[10]:


# checks for duplicate values
dataset.duplicated().sum()


# In[11]:


dataset = dataset[['Level','Salary']]


# In[12]:


# # Handle Missing data
# def handle_min_values(dataset):
#     # replace min values by mean
#     dataset.replace(0, dataset.mean(), inplace=True)
#     return dataset

# dataset = handle_min_values(dataset)


# In[13]:


# #check dataset replace with mean or not
# dataset.min()


# In[14]:


# # seperate fetures and label
# x_data = dataset.iloc[:, :-1].values
# y_data = dataset.iloc[:, 1].values


# In[15]:


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


# In[16]:


# #convert numpy.ndarray to DataFrame
# x_data = pd.DataFrame(x_data)
# x_data.shape


# In[17]:


# create directory to store csv files
# os.mkdir("CSV_files")


# In[18]:


# split dataset 

def splitdata(dataset):
    # split train and test data
    train, test = train_test_split(dataset,test_size = 0.20, random_state=0)
    print("train : ", train.shape, " test : ", test.shape)

    # saving datasets into csv files
    test.to_csv('CSV_files/test_file.csv',index=False,encoding='utf-8')

    # divide train data into train and cross validation 
    train_data, crossV_data = train_test_split(train,test_size = 0.30,random_state=0)
    
     #load data into csv for train and cross validation
    train_data.to_csv('CSV_files/train_file.csv',index=False,encoding='utf-8')
    crossV_data.to_csv('CSV_files/CValidation_file.csv',index=False,encoding='utf-8')
    
    print("train_data : ", train_data.shape, " crossV_data : ", crossV_data.shape)

splitdata(dataset)


# In[19]:


# load dataset
train_dataset = pd.read_csv ("CSV_files/train_file.csv")
print("Dataset has {} rows and {} Columns".format(train_dataset.shape[0],train_dataset.shape[1])) 


# In[20]:


train_dataset.head()


# In[33]:


x_train = train_dataset.iloc[:,:-1].values
y_train = train_dataset.iloc[:,1].values  
    


# In[41]:


# fitting simple linear regression model to the training dataset
# lin_reg = LinearRegression(normalize=True)  
# lin_reg.fit( x_train, y_train)  

# fitting polynomial regression model to the training dataset
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x_train)
# fit into multiple Linear regression model
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)


# In[47]:


class Polynomial_Reg():
    
    def reference_module(self,x_train):
        # fitting polynomial regression model to the training dataset
        poly_reg = PolynomialFeatures(degree=5)
        x_poly = poly_reg.fit_transform(x_train)
        return poly_reg, x_poly
      

    def create_module(self,x_train,y_train, x_poly):
        # fit into multiple Linear regression model
        lin_reg2 = LinearRegression()
        lin_reg2.fit(x_poly,y_train)
        return lin_reg2
    
    def create_piklefile(self,poly_reg, lin_reg2):
        fileObject = open("train_data.pkl",'wb')       
        # dump train model pickle file
        file = open('Polynomial_RegModule.pkl', 'wb')
        pickle.dump(poly_reg,file)
        pickle.dump(lin_reg2,file)
        # here we close the fileObject
        file.close()          
        
    
    def y_prediction(self,x_train,lin_reg2,poly_reg):
        # predicting the train set result
        y_pred_train = lin_reg2.predict(poly_reg.fit_transform(x_train))
        return y_pred_train
    
    def accuracy(self,y_predict_train,y_train):
        # accuracy using r2 score
        acc_r2 = r2_score(y_train, y_predict_train)*100      
#         acc_r2 = (1-error)*100
  
        total_error = mean_absolute_error(y_train, y_predict_train)
        mean_ab=( 1- (total_error / len(y_train))) *100
        
        mean_sq  = mean_squared_error(y_train, y_predict_train) 

        mean_sq_log = mean_squared_log_error(y_train, y_predict_train)  
    
        median_ab_error = median_absolute_error(y_train, y_predict_train)
        
        return acc_r2,mean_ab,mean_sq,mean_sq_log, median_ab_error
    

    
    def visualization(self,x_train,y_train,poly_reg, lin_reg2):
        # Visualization the Decision Tree result (for higher resolution & smoother curve)
        x_grid=np.arange(min(x_train),max(x_train),0.1)
        x_grid=x_grid.reshape((len(x_grid),1))
        
        plt.scatter(x_train,y_train,color='red')
        plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
        plt.title('predict salary  based on position (Training Set)')
        plt.xlabel('Level')
        plt.ylabel('Salary')
        plt.show()
        

def main():
    #class obj created
    obj  = Polynomial_Reg()
    
    # seperate fetures and label
    # here we taking only 2 columns level and salary
    x_train = train_dataset.iloc[:,:-1].values
    y_train = train_dataset.iloc[:,1].values  
    
    poly_reg, x_poly = obj.reference_module(x_train)
    
    lin_reg2 = obj.create_module(x_train,y_train, x_poly)
#     print("\nModule created")
    
    obj.create_piklefile(poly_reg, lin_reg2)
#     print("\nPikle file created")
  
    y_train_pre = obj.y_prediction(x_train,lin_reg2, poly_reg)
#     print("\n\n y_prediction:",y_train_pre)
    
    acc_r2,mean_ab,mean_sq,mean_sq_log, median_ab_error = obj.accuracy(y_train_pre,y_train)
    print("\n Accuracy train by acc_r2", acc_r2)
    print("\n Accuracy train by mean_ab", mean_ab)
    print("\n Accuracy train by mean_sq", mean_sq)
    print("\n Accuracy train by mean_sq_log", mean_sq_log)
    print("\n Accuracy train by median_ab_error", median_ab_error)
    
    
    
    obj.visualization(x_train,y_train, poly_reg, lin_reg2)

if __name__ == '__main__':
    main()


# In[43]:


# Cross Validation

# load dataset
CV_dataset = pd.read_csv ("CSV_files/CValidation_file.csv")
print("Dataset has {} rows and {} Columns".format(CV_dataset.shape[0],CV_dataset.shape[1])) 


# In[44]:


#     #cross validation
# file1 = open('SimpleLRModulefile.pkl', 'rb')
# reg1 = pickle.load(file1)


# In[45]:


class Cross_validation():
           
    def y_prediction(self,x_train,lin_reg2,poly_reg):
        # predicting the train set result
        y_pred_train = lin_reg2.predict(poly_reg.fit_transform(x_train))
        return y_pred_train
        
#         # predicting the test set result
#         return regression.predict(x_train)
    
    def accuracy(self,y_predict_train,y_train):
        # acc using r2
        acc_r2 = r2_score(y_train, y_predict_train)*100
#         acc_r2 = (1-error)*100
        
        # using median_ab_error
        median_ab_error = median_absolute_error(y_train, y_predict_train)
        return acc_r2, median_ab_error
    
    def visualization(self,x_cv,y_cv,poly_reg, lin_reg2):
        # visualizing the testing set result
        x_grid=np.arange(min(x_cv),max(x_cv),0.1)
        x_grid=x_grid.reshape((len(x_grid),1))
        plt.scatter(x_cv,y_cv,color='red')
        plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
        plt.title('predict salary  based on position (Cross Validation Set)')
        plt.xlabel('Level')
        plt.ylabel('Salary')
        plt.show()
        

def main():
    #class obj created
    obj  = Cross_validation()
    
    # seperate fetures and label
    x_cv = CV_dataset.iloc[:,:-1].values
    y_cv = CV_dataset.iloc[:,1].values
 
    #     print(x_cv.shape,y_cv.shape)
    #cross validation
    file1 = open('Polynomial_RegModule.pkl', 'rb')
    reg1 = pickle.load(file1)
    
    # y_prediction ( cross validation)   
    y_cv_pre = obj.y_prediction(x_cv,lin_reg2,poly_reg)
    print("\n\n y_prediction:",y_cv_pre)
    
    acc_r2, median_ab_error= obj.accuracy(y_cv_pre,y_cv)
    print("\n Accuracy train by acc_r2", acc_r2)
    print("\n Accuracy train by median_ab_error", median_ab_error)

    obj.visualization(x_cv,y_cv,poly_reg, lin_reg2)

if __name__ == '__main__':
    main()


# In[ ]:


# Here decision tree gives 100% or very small accuracy bcoz of overfitting and small amount of dataset


# In[ ]:




