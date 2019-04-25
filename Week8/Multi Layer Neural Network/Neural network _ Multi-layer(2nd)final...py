#!/usr/bin/env python
# coding: utf-8

# In[3]:



import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt  


# In[4]:


#load dataset
df_original= pd.read_csv("Churn_Modelling.csv")
df = df_original
# show top 5 rows of dataset
df.head()


# In[5]:


print("Dataset has {} rows and {} Columns".format(df.shape[0],df.shape[1])) 


# In[6]:


df.dtypes


# In[7]:


# check dataset information
df.info()


# In[8]:


df.duplicated().sum()


# In[9]:


df.describe()


# In[10]:


#check for duplicate values
df.duplicated().sum()


# In[11]:


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


# In[12]:


df.describe().T


# In[13]:


df.dtypes


# In[14]:


df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)


# In[15]:


df.shape


# In[16]:


# Handle categorical data
# df = pd.get_dummies(df)
df.dtypes


# In[17]:


df = pd.get_dummies(df)


# In[18]:


# get dummy variables whose are in categorical type
# for name in df.columns:
#     if df[name].dtype == "object":
#             df[name] = pd.get_dummies(df[name]) 


# In[19]:


# check the correlation
corr = df.corr()
sb.heatmap(corr)


# In[20]:


df.shape


# In[21]:


sb.countplot(x='Exited',  data=df, palette='hls')
plt.show()


# In[22]:


def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df
df = Feature_Scaling(df)


# In[23]:


# seperate data set
def features(df):
    y_new = df.Exited
#     y.head()
    df = df.drop('Exited', axis = 1)
    print("Seperate Exited column from dataset")
    return df, y_new
df, y_new = features(df)


# In[24]:


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


# In[31]:


class NueralNetwork:
    def __init__(self):
        # assign learning rate
        self.learning_rate = 0.0070
        self.epoch = 1000
        
    def gradientDescent(self, x_train_data,y_train_data, w, b):
        m = len(x_train_data) 
        layers = (x_train_data.shape[1], 4, 5, 3, 1)
        
        z = [0] * len(layers)
        a = [0] * len(layers)
        dg = [0] * len(layers)
        dA = [0] * len(layers)
        dz = [0] * len(layers)
        db = [0] * len(layers)
        dw = [0] * len(layers)
        
#         A = [0] * len(layers)
        a[0] = x_train_data.T
        
        weight = []
        bias = []
        
        for i in range(1, len(layers)):
            weight.append(np.random.rand(layers[i], layers [i-1])*0.001)
            bias.append(np.zeros((layers[i], 1)) )   
            # without brackets it gives TypeError: data type not understood
            
        for j in range(self.epoch):
            for i in range(len(layers) - 1):
                print("\n\nForward : layer = ", i )
                #----------Forward Propagation ---------------
                #hypothesis function
                z[i] = np.dot(weight[i] , a[i]) + bias[i]
                print("z :", z[i].shape)
                # we are not taking a[i - 1] bcoz it taking z (4, 49) shape
                # calculate activation function (sigmoid function)
                a[i+1] = 1 / (1 + np.exp(-z[i]))
                print("a[i]",a[i+1].shape)
                # A[i] = g[i] * (z[i])  --> Activation function * hepo
                
            for i in reversed(range(len(layers) -1)):
                print("!!!!!!!!!!!!!!!!")
                #-------Backword Propogation-------------
                print("Backword : layer = ", i)
                #loss function derivation
                dA[i] = (-(y_train_data.T/ a[i+1]) + ((1 - y_train_data.T) / (1 - a[i+1])))
                print("dA[i] = ", dA[i].shape)
    
                #derivation of sigmoid function
                dg[i] = (1 / (1 + np.exp(-z[i]))) * (1 - (1 / (1 + np.exp(-z[i]))))
#                 dg[i] = a[i+1] * 1 - a[i+1] 
                print("dg[i] = ", dg[i].shape)
                
                dz[i] = dA[i] * dg[i]
                print("dz[i] = ", dz[i].shape)

                dw[i] =  np.dot(dz[i], a[i].T) / m
                print("dw[i] = ", dw[i].shape)

                db[i] = np.sum(dz[i], axis =1 ,keepdims = True) / m
                
                # update weight and bias
                weight[i] = weight[i] - np.dot(self.learning_rate, dw[i])
                bias[i] = bias[i] - np.dot(self.learning_rate, db[i])

        
        return [weight, bias] 
    
    
    def predict(self, x_test_data, parameters):
            #reshape
            a = [0] * 5
            a[0] = x_test_data.T
            z = [0] * 5
            a = [0] * 6
            for i in range(4):  
                z[i] = np.dot(parameters[0][i], a[i]) + parameters[1][i]
                a[i+1] = 1 / (1 + np.exp(-z[i])) 
            return a[-1]
            return y_predict

    def accuracy(self, y_test_data, y_pred_test):
        y_pred_test = np.nan_to_num(y_pred_test)
   
        test_accuracy = 100 - (np.mean(np.abs(y_pred_test - y_test_data)) * 100)        
        return test_accuracy

        
def main(x_train_data, y_train_data, x_test_data, y_test_data):
    # class object created
    obj = NueralNetwork()
    
    print("/nx_train_data shape",x_train_data.shape)
    print("x_test_data shape",x_train_data.shape)
    
    # intialization
    x_col = 48
    
    # create vector for theta's(weights)
    w = np.full((x_col + 1, 1),0.5)
    b = np.zeros((1, 1), dtype = 'float')
    
    x_train_data = np.column_stack((np.ones((x_train_data.shape[0], 1)), x_train_data))
    x_test_data = np.column_stack((np.ones((x_test_data.shape[0], 1)), x_test_data))
    print("\n\nx_train_data shape",x_train_data.shape)
    print("x_test_data shape",x_train_data.shape)
    
    parameters = obj.gradientDescent(x_train_data, y_train_data, w, b)

    y_predict_test = obj.predict(x_test_data, parameters)
#     print("y_predict test:",y_predict_test.shape)

    y_predict_train = obj.predict(x_train_data, parameters)
#     print("y_predict train:",y_predict_train.shape)

    train_accuracy=obj.accuracy(y_train_data, y_predict_train)
    test_accuracy=obj.accuracy(y_test_data, y_predict_test)

    print("Accuracy train:", train_accuracy)
    print("Accuracy test:",  test_accuracy)

    
if __name__ == '__main__':
    main(x_train_data, y_train_data, x_test_data, y_test_data)


# In[ ]:




