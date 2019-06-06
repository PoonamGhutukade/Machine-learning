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
    
    def print_data(para):
        print("Hello!!!",para)
        return
    
        # split dataset 
    def splitdata(dataset):
        # split train and test data
        train, test = train_test_split(dataset,test_size = 0.30, random_state=0)
        print("train : ", train.shape, " test : ", test.shape)

        # saving datasets into csv files
        test.to_csv('CSV_files/test_file.csv',index=False,encoding='utf-8')

        # divide train data into train and cross validation 
        train_data, crossV_data = train_test_split(train,test_size = 0.30,random_state=0)

         #load data into csv for train and cross validation
        train_data.to_csv('CSV_files/train_file.csv',index=False,encoding='utf-8')
        crossV_data.to_csv('CSV_files/CValidation_file.csv',index=False,encoding='utf-8')

        print("train_data : ", train_data.shape, " crossV_data : ", crossV_data.shape)

