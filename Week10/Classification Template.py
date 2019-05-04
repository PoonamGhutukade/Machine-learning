# import libraries

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

    #Classification library
    from sklearn.linear_model import LogisticRegression
    #confusion matix
    from sklearn.metrics import confusion_matrix 
    #visualisation
    from matplotlib.colors import ListedColormap

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

    from util import Util_class as obj_util
#-----------------------------------------------------------------
    
# load dataset
dataset_original = pd.read_csv ("dataset.csv")
dataset = dataset_original
dataset.head()

print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 

# check dataset information
dataset.info()

# descibe the dataset
dataset.describe().T

# handling missing data if nessesary
"""
if missing values are present
imputer = Imputer(missing_values=0, axis=0)
imputer = imputer.fit(x_data[:, 3:16])
"""
dataset.isnull().sum()

# check for duplicate values
dataset.duplicated().sum()


#Check categpries for categorical data
dataset.Gender[:5]

#--------------------------------------------------

# create directory to store csv files
os.mkdir("CSV_files")

#split dataset into train, test and cross validation also save csv files
obj_util.splitdata(dataset, 0.30, 0.25)

# load train dataset
train_dataset = pd.read_csv ("CSV_files/train_file.csv")
print("Dataset has {} rows and {} Columns".format(train_dataset.shape[0],train_dataset.shape[1])) 

# seperate fetures and label

#train dataset
# features -> age and estimated salary
x_train = train_dataset.iloc[:,[2,3]].values
# label -> purchased
y_train = train_dataset.iloc[:,4].values  
y_train = y_train.reshape(-1,1)
print("x_train :",x_train.shape,"& y_train:",y_train.shape)

#data Preprocessing
train_dataset.info()

# seperate fetures and label

# features -> age and estimated salary
x_train = train_dataset.iloc[:,[2,3]].values
# label -> purchased
y_train = train_dataset.iloc[:,4].values  
y_train = y_train.reshape(-1,1)
print("x_train :",x_train.shape,"& y_train:",y_train.shape)

# if needed
# handle categorical data
def handle_categorical_data(x_data):
    #encode categorical data
    label_encod = LabelEncoder()
    x_data[:, 1] = label_encod.fit_transform(x_data[:, 1])
    
    # one hot encoding
    onehotencode = OneHotEncoder(categorical_features= [1])
    x_data = onehotencode.fit_transform(x_data).toarray()
    
    return x_data
    
x_data = handle_categorical_data(x_data)

#feature scalling (here data will be converted into float)
def feature_scalling(x_train,y_train):
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    return sc_x, x_train
    
sc_x, x_train = feature_scalling(x_train,y_train)


# #Adjust according to code
# #feature scalling (here data will be converted into float)
# def feature_scalling(x_data_set,y_data_set):
#     sc_x = StandardScaler()
#     sc_y = StandardScaler()

#     x = sc_x.fit_transform(x_data_set.reshape(-1, 1))
#     y = sc_y.fit_transform(y_data_set.reshape(-1, 1))
    
#     return x, y, sc_x, sc_y
    
# x, y, sc_x, sc_y = feature_scalling(x_data_set,y_data_set)


# load dataset for Cross Validation
CV_dataset = pd.read_csv ("CSV_files/CValidation_file.csv")
print("Dataset has {} rows and {} Columns".format(CV_dataset.shape[0],CV_dataset.shape[1])) 


#for cross validation
# features -> age and estimated salary
x_cv = CV_dataset.iloc[:,[2,3]].values
# label -> purchased
y_cv = CV_dataset.iloc[:,4].values  
y_cv = y_cv.reshape(-1,1)
print("x_cv :",x_cv.shape,"& y_cv:",y_cv.shape)