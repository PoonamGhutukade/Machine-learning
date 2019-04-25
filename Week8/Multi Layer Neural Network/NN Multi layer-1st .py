#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt  


# In[123]:


#load dataset
df_original= pd.read_csv("bank.csv", delimiter=';')
df = df_original
# show top 5 rows of dataset
df.head()


# In[124]:


# check dataset information
df.info()


# In[125]:


df['job'].unique()


# In[126]:


len(df['job'].unique())


# In[127]:


print("Dataset has {0} rows & {1} columns ".format(df.shape[0], df.shape[1]))


# In[128]:


# check for the duplicates
df.duplicated().sum()


# In[129]:


"""Handling Missing Data"""
def missing_data(df):
     
        # check null values in each column
        print("\nNull values in dataset:\n",df.isnull().sum())
        
        #check for minimum values
        print("\n\nMin values:\n",  df.min())
        
#         df.replace(np.NaN, df.mean, inplace = True)
        
#         print("\n After replacing minimum values\n",df.min())
        
        # check for duplicate data
#         df.duplicated().sum()
#         print("\nCheck duplicate values:\n",df.duplicated().sum())
missing_data(df)


# In[130]:


# descibe our data
df.describe().T


# In[131]:


# df['y']


# In[132]:


# get dummy variables
# df['y_dummy'] = df.y.map({'yes':1, 'no':0})

df.replace(['yes', 'no'],[1,0],inplace= True)
df.head()


# In[133]:


# def check_skew(df):
#         """If skewness value lies above +1 or below -1, data is highly skewed. 
#         If it lies between +0.5 to -0.5, it is moderately skewed. 
#         If the value is 0, then the data is symmetric"""
        
#         print("\n Mean: \n",df.mean(), "\n\nSkew : \n",df.skew(), "\n\nMedian: \n", df.median())
#         df.hist()

# check_skew(df) 


# In[134]:


df = pd.get_dummies(df)
df.head()


# In[135]:


corr = df.corr()
sb.heatmap(corr)


# In[136]:


df.corr()


# In[137]:


# df['y']


# In[138]:


# check skewness for target variable
#         """If skewness value lies above +1 or below -1, data is highly skewed. 
#         If it lies between +0.5 to -0.5, it is moderately skewed. 
#         If the value is 0, then the data is symmetric"""

# sb.distplot(df['y'])
# print ("Skewness of y is {}".format(df['y'].skew()))

# print("\n Mean: \n",df['y'].mean(), "\n\nSkew : \n",df['y'].skew(), "\n\nMedian: \n", df['y'].median())
#         df.hist()


# In[139]:


# print("\nSkewness for y")
# df['y'] = np.sqrt(df['y'])
# print('Skewness is :', df['y'].skew())
# sb.distplot(df['y'])
# plt.show()
# # show() function remove next line--> "<matplotlib.axes._subplots.AxesSubplot at 0x7f9a114a0fd0>"


# In[140]:


#remove skewness
def remove_skew(df):
    print("\nSkewness for y")
    df['y'] = np.cbrt(df['y'])
    print('Skewness is :', df['y'].skew())
    sb.distplot(df['y'])
    plt.show()
# remove_skew(df)


# In[141]:


print (corr['y'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['y'].sort_values(ascending=False)[-10:]) #last 5 values`


# In[ ]:





# In[142]:


#now transforming the target variable
def remove_skew():
    print("\nSkewness for y")
#         df['x2'] = (np.square(df['x2']))
    df['y'] = np.log(df['y'])
    print('Skewness is :', df['y'].skew())
    sb.distplot(df['y'])
    plt.show()
# target = np.log(df['y'])
# print ('Skewness is', target.skew())
# sb.distplot(target)
# remove_skew()


# In[143]:


# check minimum values in each column
df.min()


# In[144]:


#replace all zeroes with mean of that column
df.replace(0, df.mean(), inplace = True)


# In[145]:


df.min()


# In[146]:


# check for ouliers
df.boxplot(rot=45, figsize=(20,5))
# sb.boxplot(data=df )
plt.show()


# In[147]:


def check_outliers():
    # Create a figure instance, and the two subplots
    fig = plt.figure()
       # here we show 4 axes
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1, figsize=(7, 20))

    sb.boxplot(df['age'], ax = ax1)
    ax1.set_title("Temprature outliers")
       
    sb.boxplot(df['balance'], ax = ax2)
    ax2.set_title("Humidity outliers ")
       
    sb.boxplot(df['duration'], ax = ax3)
    ax3.set_title(" Wind Speed outliers ")
       
    sb.boxplot(df['pdays'], ax = ax4)
    ax4.set_title("Wind Bearing  outliers ")
       
    sb.boxplot(df['previous'], ax = ax5)
    ax5.set_title("Visibility outliers ")
       
    sb.boxplot(df['y'], ax = ax6)
    ax6.set_title("Pressure outliers ")
       
    plt.subplots_adjust(hspace=1)
    plt.show()
    
check_outliers()


# In[148]:


def remove_outlier(df):
       
        low = .2
        high = .75
        quant_df = df.quantile([low, high])
        for name in list(df.columns):
            if ptypes.is_numeric_dtype(df[name]):
                df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
                sb.boxplot(df)
remove_outlier(df)


# In[149]:


# check for ouliers
df.boxplot(rot=45, figsize=(20,5))
# sb.boxplot(data=df )
plt.show()


# In[150]:


print('Head for df')
df.head()


# In[151]:


def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df
df = Feature_Scaling(df)


# In[152]:


# seperate data set
def features(df):
    y_new = df.y
#     y.head()
    df = df.drop('y', axis = 1)
    print("Seperate Exited column from dataset")
    return df, y_new
df, y_new = features(df)


# In[153]:


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


# In[164]:


class NueralNetwork:
    def __init__(self):
        # assign learning rate
        self.learning_rate = 0.09800
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
        
        a[0] = x_train_data.T
        
        weight = []
        bias = []
        
        for i in range(1, len(layers)):
            weight.append(np.random.rand(layers[i], layers [i-1])*0.001)
            bias.append(np.zeros((layers[i], 1)) )   
            # without brackets it gives TypeError: data type not understood
            
        for j in range(self.epoch):
            for i in range(len(layers) - 1):
                
                #----------Forward Propagation ---------------
                
                #hypothesis function
                z[i] = np.dot(weight[i] , a[i]) + bias[i]
               
                # we are not taking a[i-1] bcoz it taking z (4, 49) shape
                # calculate activation function (sigmoid function)
                a[i+1] = 1 / (1 + np.exp(-z[i]))
               
                
            for i in reversed(range(len(layers) -1)):

                #-------Backword Propogation-------------

                #loss function derivation
                dA[i] = (-(y_train_data.T/ a[i+1]) + ((1 - y_train_data.T) / (1 - a[i+1])))
#                 print("dA[i] = ", dA[i].shape)
    
                #derivation of sigmoid function
                dg[i] = (1 / (1 + np.exp(-z[i]))) * (1 - (1 / (1 + np.exp(-z[i]))))
#                 print("dg[i] = ", dg[i].shape)
                
                dz[i] = dA[i] * dg[i]
#                 print("dz[i] = ", dz[i].shape)

                dw[i] =  np.dot(dz[i], a[i].T) / m
#                 print("dw[i] = ", dw[i].shape)

                db[i] = np.sum(dz[i], axis =1 ,keepdims = True) / m
                
                # update weight and bias
                weight[i] = weight[i] - np.dot(self.learning_rate, dw[i])
                bias[i] = bias[i] - np.dot(self.learning_rate, db[i])
                
        print("shape of Z", z[i].shape)
        print("shape of a[i+1]", a[i+1].shape)
        print("shape of dg[i]", dg[i].shape)
        print("shape of da[i]", dA[i].shape)
        print("shape of dz[i]", dz[i].shape)
        print("shape of dw[i]", dw[i].shape)
        print("shape of db[i]", db[i].shape)
        print("shape of weight[i]", weight[i].shape)
        print("shape of baised[i]", bias[i].shape)
        
        return [weight, bias] 
    
    
    
    def predict(self, x_test_data, parameters):
            i = 0
            z = [0] * 5
            a = [0] * 5
            a[0] = x_test_data.T
            for i in range(4):
                z[i] = np.dot(parameters[0][i], a[i]) + parameters[1][i]
                a[i+1] = 1 / (1 + np.exp(-z[i])) 
                #predict y value
            return a[-1]

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
    
#     alpha --> self.l_rate = 0.001
#     Accuracy train: 75.49879331841993
#     Accuracy test: 75.30290492969557
    
if __name__ == '__main__':
    main(x_train_data, y_train_data, x_test_data, y_test_data)


# In[ ]:


# alpha = 0.0001

# Accuracy train: 51.223403048474005
# Accuracy test: 51.21395614615359
    
# alpha = 0.001    
# Accuracy train: 58.81136212550142
# Accuracy test: 58.74346137200593
    
# alpha = 0.0070  
# Accuracy train: 73.35175547678608
# Accuracy test: 73.17242555145376
    
# alpha = 0.0080 
# Accuracy train: 74.20971477633549
# Accuracy test: 74.0237750537206
    
# alpha = 0.080 
# Accuracy train: 79.54060025016788
# Accuracy test: 79.3116858320872


# In[165]:


############### Following code just for cheking shapes of each step


# In[166]:


############### Following code just for cheking shapes of each step
class NueralNetwork:
    def __init__(self):
        # assign learning rate
        self.learning_rate = 0.0070
        self.epoch = 60000
        
    def gradientDescent(self, x_train_data,y_train_data, w, b):
        m = len(x_train_data) 
        layers = (x_train_data.shape[1], 4, 5,  3, 1)
        
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
        bais = []
        
        for i in range(1, len(layers)):
            weight.append(np.random.rand(layers[i], layers [i-1])*0.001)
            bais.append(np.zeros(((layers[i], 1))) )
       
        for j in range(self.epoch):
            for i in range(0, len(layers) - 1):
                print("\n\nLayer : ", i)
                #----------Forward Propagation ---------------
                print("weight ", i, "= ", weight[i].shape)

                #hypothesis function
                z[i] = np.dot (weight[i] , a[i]) + bais[i]
                #  z[i] = np.dot (weight[i] , A[i - 1]) + bais[i] 
                # we are not taking a[i - 1] bcoz it taking z (4, 49) shape
#                 print("z[i] = weight",weight[i].shape, "A[i] = ",A[i].shape, "bais[i] = ", bais[i].shape )
                print("z (hypo)",i, "= ",z[i].shape)
                
                # calculate activation function (sigmoid function)
                a[i+1] = 1 / (1 + np.exp(-z[i]))
                print("a",i, "= ",a[i+1].shape)
                # A[i] = g[i] * (z[i])  --> Activation function * hepo
#                 A[i] = a[i] * z[i]
                
                #-------Backword Propogation-------------
                
                #loss function derivation
#                 da[i] = (-(y_train_data.T / a[i+1]) + ((1 - y_train_data.T) / (1 - a[i+1])))
#                 print("dA[i] = ",y_train_data.shape, a[i].shape)
                dA[i] = (-(y_train_data.T/ a[i+1]) + ((1 - y_train_data.T) / (1 - a[i+1])))

                print("dA[i] = ", dA[i].shape)
    
                #derivation of sigmoid function
                dg[i] = a[i+1] * 1 - a[i+1] 
                print("dg[i] = ", dg[i].shape)
                
                dz[i] = dA[i] * dg[i]
                print("dz[i] = ", dz[i].shape)

                dw[i] =  np.dot(dz[i], a[i].T) / m
                print("dw[i] = ", dw[i].shape)


                # ---------------------error here put dp[i] tp solve it----------------------------------
                db = np.sum(dz[i], axis = 1 ,keepdims = True) / m
#                 print("db[i] = ", db[i].shape)
                
                # update weight and bias
                # w = w - alpha * dw
                # b = b - alpha * db

                weight[i] = weight[i] - np.dot(self.learning_rate, dw[i])
                bais[i] = bais[i] - np.dot(self.learning_rate, db[i])
            
        print(" z shape :", z.shape)
        print("sigmoid :", a.shape)
        print("dz :", dz.shape)
        print("dw :",dw.shape)
        print("db :", db.shape)
        print("weight :", weight.shape)
        print("Bias :", bias.shape)  
        return [weight, bais] 
    
    
    def predict(self, x_test_data, parameters):
            #reshape
            a = [0] * 6
            a[0] = x_test_data.T
            z = [0] * 6
#             a = [0] * 6
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
    print("y_predict test:",y_predict_test.shape)
    
    y_predict_train = obj.predict(x_train_data, parameters)
    print("y_predict train:",y_predict_train.shape)
    
    train_accuracy=obj.accuracy(y_train_data, y_predict_train)
    test_accuracy=obj.accuracy(y_test_data, y_predict_test)

    print("Accuracy train:", train_accuracy)
    print("Accuracy test:",  test_accuracy)
    
    
if __name__ == '__main__':
    main(x_train_data, y_train_data, x_test_data, y_test_data)


# In[ ]:




