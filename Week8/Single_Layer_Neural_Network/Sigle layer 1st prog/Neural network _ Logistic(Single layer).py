#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt  


# In[139]:


#load dataset
df_original= pd.read_csv("bank.csv", delimiter=';')
df = df_original
# show top 5 rows of dataset
df.head()


# In[140]:


print("Dataset has {} rows and {} Columns".format(df.shape[0],df.shape[1])) 


# In[141]:


# check dataset information
df.info()


# In[142]:


# checking for largte category
print(df['job'].unique())


# In[143]:


len(df['job'].unique())


# In[144]:


# check for null values
df.isnull().sum()


# In[145]:


# check for the duplicate values
df.duplicated().sum()


# In[146]:


# descibe our data
df.describe().T
# what is previous column here.. 75 % min values are zero


# In[147]:


# print("\nDrop unwanted columns")
# df = df.drop(['marital','education','contact','poutcome', 'day', 'month'], axis = 1)
# df.head()


# In[148]:


# check datatypes for each column
df.dtypes


# In[149]:


# def check_skew(df):
#         """If skewness value lies above +1 or below -1, data is highly skewed. 
#         If it lies between +0.5 to -0.5, it is moderately skewed. 
#         If the value is 0, then the data is symmetric"""
        
#         print("\n Mean: \n",df.mean(), "\n\nSkew : \n",df.skew(), "\n\nMedian: \n", df.median())
#         df.hist()

# check_skew(df) 


# In[150]:


# get dummy variables
# df['y_dummy'] = df.y.map({'yes':1, 'no':0})

# handle the categiricol data
df.replace(['yes', 'no'],[1,0],inplace= True)
df.head()


# In[151]:


# get dummies from dataset
df = pd.get_dummies(df)
df.head()


# In[152]:


#chekc correlation of target "y" with each other data
corr = df.corr()
sb.heatmap(corr)


# In[153]:


df.corr()


# In[154]:


# test dataset for all values?(e.g min, max,mean etc)
df.describe()


# In[155]:


# large number of data with min zero, hence replace those values with mean of that column
df.replace(0.0, df.mean(),inplace = True)


# In[156]:


#agin validate the dataset, to min values (changing or not)
df.describe()


# In[157]:


# easy way to check correlation of target variable to all oather features after getdummies
print (corr['y'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['y'].sort_values(ascending=False)[-10:]) #last 5 values`


# In[158]:


def check_skew(df):
        """If skewness value lies above +1 or below -1, data is highly skewed. 
        If it lies between +0.5 to -0.5, it is moderately skewed. 
        If the value is 0, then the data is symmetric"""
        
        print("\n Mean: \n",df['y'].mean(), "\n\nSkew : \n",df['y'].skew(), "\n\nMedian: \n", df['y'].median())
        sb.distplot(df['y'])
        plt.show()

check_skew(df)


# In[159]:


# def remove_skew(df):
#         print("\nRemove Skewness ")
#         df['y'] = (np.log(df['y']))
#         print("\n Mean: \n",df['y'].mean(), "\n\nSkew : \n",df['y'].skew(), "\n\nMedian: \n", df['y'].median())
#         sb.distplot(df['y'])
# remove_skew(df)


# In[160]:


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


# In[161]:


# check for ouliers
df.boxplot(rot=45, figsize=(20,5))
# sb.boxplot(data=df )
plt.show()


# In[162]:


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


# In[163]:


def remove_outlier(df):
       
        low = .20
        high = .85
        quant_df = df.quantile([low, high])
        for name in list(df.columns):
            if ptypes.is_numeric_dtype(df[name]):
                df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
                sb.boxplot(df)
remove_outlier(df)


# In[164]:


# check for ouliers
df.boxplot(rot=45, figsize=(20,5))
# sb.boxplot(data=df )
plt.show()


# In[165]:


# plot y to check how much previous client subscribed to term deposit
sb.countplot(x='y',  data=df, palette='hls')
plt.show()


# In[166]:


# # Separating the output and the parameters data frame
# def separate(df):
#     y_new = df.y
#     print("Seperate y column from dataset")
#     df = df.drop('y', axis=1)
#     return df, y_new
# df, y_new = separate(df)


# In[167]:


# do feature scale by min max normalization
def feature_scaling(df):
#     print("\n By Z score Method(Standerdization)  ")
#     df = np.divide((df - df.mean()),df.std())
    for name in df.columns:
        df[name] = (df[name] - df[name].min()) / (df[name].max()-df[name].min())
    print(df.head())
    return df
df = feature_scaling(df) 


# In[168]:


# Separating the output and the parameters data frame
def separate(df):
    y_new = df.y
    print("Seperate y column from dataset")
    df = df.drop('y', axis=1)
    return df, y_new
df, y_new = separate(df)


# In[169]:


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


# In[170]:


print('Head for df')
df.head()


# In[171]:


print('Head for new Y')
y_new.head()


# In[172]:


class NueralNetwork_Logistic:
    def __init__(self):
        # assign learning rate
        self.learning_rate = 0.000000000070
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
    x_col = 48
    
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

    print("Accuracy train:", train_accuracy)
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





# In[ ]:




