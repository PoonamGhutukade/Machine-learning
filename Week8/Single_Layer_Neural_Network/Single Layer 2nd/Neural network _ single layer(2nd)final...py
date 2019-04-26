#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt  


# In[27]:


#load dataset
df_original= pd.read_csv("Churn_Modelling.csv")
df = df_original
# show top 5 rows of dataset
df.head()


# In[28]:


print("Dataset has {} rows and {} Columns".format(df.shape[0],df.shape[1])) 


# In[29]:


df.dtypes


# In[30]:


# check dataset information
df.info()


# In[31]:


df.duplicated().sum()


# In[32]:


df.describe()


# In[33]:


#check for duplicate values
df.duplicated().sum()


# In[34]:


"""Handling Missing Data"""
def missing_data(df):
     
        # check null values in each column
        print("\nNull values in dataset:\n",df.isnull().sum())
        
        #check for minimum values
        print("\n\nMin values:\n",  df.min())
        
        df.replace(np.NaN, df.mean, inplace = True)
        
        print("\n After replacing minimum values\n",df.min())
        
        # check for duplicate data
#         df.duplicated().sum()
#         print("\nCheck duplicate values:\n",df.duplicated().sum())
missing_data(df)


# In[35]:


df.describe().T


# In[11]:


df.dtypes


# In[12]:


df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)


# In[13]:


df.shape


# In[14]:


# Handle categorical data
# df = pd.get_dummies(df)
df.dtypes


# In[16]:


df = pd.get_dummies(df)


# In[17]:


# get dummy variables whose are in categorical type
# for name in df.columns:
#     if df[name].dtype == "object":
#             df[name] = pd.get_dummies(df[name]) 


# In[18]:


# check the correlation
corr = df.corr()
sb.heatmap(corr)


# In[19]:


df.shape


# In[20]:


sb.countplot(x='Exited',  data=df, palette='hls')
plt.show()


# In[21]:


def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df
df = Feature_Scaling(df)


# In[22]:


# seperate data set
def features(df):
    y_new = df.Exited
#     y.head()
    df = df.drop('Exited', axis = 1)
    print("Seperate Exited column from dataset")
    return df, y_new
df, y_new = features(df)


# In[23]:


def split_data(df):
    # 70 % training datset
    train_per = int(0.70*len(df))
    print("Train dataset:", train_per)
    # 30% test dataset
    test_per = len(df)-train_per
    print("Test dataset:", test_per)
    
    print("\nConvert pandas datafrem into numpy")
    x_train_data = np.array(df[:train_per])   
    x_test_data  = np.array(df[:test_per])
    
    
    train_per_y = int(0.70*len(y_new))
    test_per_y = len(y_new)-train_per_y

    
    y_train_data = np.array(y_new[:train_per_y])
    y_test_data = np.array(y_new[:test_per_y])
    
    #reshpe (1357,) to (1357,1) for train and test dataset
    y_test_data = y_test_data.reshape(-1,1)
    y_train_data = y_train_data.reshape(-1,1)
        
    
    print("\nX train data shape:", x_train_data.shape)
    print("y train data shape:", y_train_data.shape)
    print("\nX test data shape:", x_test_data.shape)
    print("y test data shape:", y_test_data.shape)

    return x_train_data, y_train_data, x_test_data, y_test_data

x_train_data, y_train_data, x_test_data, y_test_data = split_data(df)        


# In[25]:


class NueralNetwork_Logistic:
    def __init__(self):
        # assign learning rate
        self.learning_rate = 0.70
        self.epoch = 60000
        
    def gradientDescent(self, x_train_data,y_train_data, w, b):
        m = len(x_train_data) 
        
        for row in range(self.epoch):
            #hypothesis function
            z = np.dot(w.T, x_train_data.T) + b
            # calculate sigmoid function
            sigmoid = 1 / (1 + np.exp(-z))
            
            #remove explicit loop and use inbuilt function of numpy 
            dz = sigmoid - y_train_data.T
            temp = np.dot(x_train_data.T, dz.T) 
#             dw = (1 / m ) * temp
            dw = temp / m
            temp1 = np.sum(dz)
            db = np.dot((1 / m) , temp1)
            db = temp1 / m 

            # update weight and bias
            w = w - (self.learning_rate * dw)
            b = b - (self.learning_rate * db)
            
        print(" z shape :", z.shape)
        print("sigmoid :", sigmoid.shape)
        print("dz :", dz.shape)
        print("dw :",dw.shape)
        print("db :", db)
        print("weight :", w.shape)
        print("Bias :", b.shape)
              
            
        return w, b 
    
    def prediction(self, x_data, w, b):
        print("$$$$$$$$$")
        print("x_data shape...",x_data.shape)
        y_prediction = np.zeros((x_data.shape[0], 1), dtype=float)
#         y_pre = np.zeros((x_data.shape[0], 1), dtype=float)
        
        # hypothesis function
        z = np.dot(x_data, w) + b
        # calculate sigmoid function
        sigmoid = 1 / (1 + np.exp(-z))
 
        # assign decision boundaries 
        for i in (range(0, len(sigmoid))):
            if round(sigmoid[i][0],2) <= 0.5:
                y_prediction[i][0] = 0
            else:
                y_prediction[i][0] = 1
        return y_prediction
    
    def accuracy(self, y_test_data, y_pred):
        count=0
        for i in range(0,len(y_test_data)):
            if y_pred[i] == y_test_data[i]:
                count = count + 1
        return count/len(y_test_data)*100
        
def main(x_train_data, y_train_data, x_test_data, y_test_data):
    # class object created
    obj = NueralNetwork_Logistic()
    print("/nx_train_data shape",x_train_data.shape)
    print("x_test_data shape",x_train_data.shape)
    
    # intialization
    x_col = 13
    
    # create vector for theta's(weights)
    w = np.full((x_col + 1, 1),0.5)
    b = np.zeros((1, 1), dtype = 'float')
    
    x_train_data = np.column_stack((np.ones((x_train_data.shape[0], 1)), x_train_data))
    x_test_data = np.column_stack((np.ones((x_test_data.shape[0], 1)), x_test_data))
    print("\n\nx_train_data shape",x_train_data.shape)
    print("x_test_data shape",x_train_data.shape)
    
    w, b = obj.gradientDescent(x_train_data, y_train_data, w, b)

    y_predict_test = obj.prediction(x_test_data, w, b)
    print("y_predict test:",y_predict_test.shape)
    
    y_predict_train = obj.prediction(x_train_data, w, b)
    print("y_predict train:",y_predict_train.shape)
    
    train_accuracy=obj.accuracy(y_train_data, y_predict_train)
    test_accuracy=obj.accuracy(y_test_data, y_predict_test)

    print("\n\nAccuracy train:", train_accuracy)
    print("Accuracy test:",  test_accuracy)
    
    """
    self.learning_rate = 0.000001
    self.epoch = 60000
    Accuracy train: 11.56763590391909
    Accuracy test: 11.864406779661017
    """
    
if __name__ == '__main__':
    main(x_train_data, y_train_data, x_test_data, y_test_data)


# In[ ]:




