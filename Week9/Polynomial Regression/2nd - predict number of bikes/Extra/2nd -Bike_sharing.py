#!/usr/bin/env python
# coding: utf-8

# In[55]:


# import libraries
import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
import pickle
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import *
from sklearn.preprocessing import normalize as N
from sklearn.preprocessing import PolynomialFeatures


# In[56]:


# read file
df_original=pd.read_csv("bike_sharing-Copy1.csv")

df =df_original
df.head()


# In[57]:


df.shape


# In[58]:


df.info()


# In[59]:


df.describe()


# In[60]:


# checks for null values
df.isnull().sum()


# In[61]:


# checks for duplicate values
df.duplicated().sum()


# In[62]:


# checks correlation with all columns
corr=df.corr()
sb.heatmap(corr)


# In[63]:


# checks correlation with all columns
print(corr['cnt'].sort_values(ascending=False)[:])


# In[64]:


df.columns


# In[65]:


df = df[['registered','cnt']]


# In[66]:


sb.boxplot(data=df)


# In[67]:


# df = preprocess_obj.remove_outlier(df)


# In[68]:


# display skewness of dataframe
target=df.skew()
sb.distplot(target)


# In[69]:


# splitting data into train & test dataset
train,test=train_test_split(df,test_size=0.2)


# In[70]:


print("train",train.shape)
print("test",test.shape)


# In[71]:


# saving datasets into csv filesS
test.to_csv('test_data.csv',index=False,encoding='utf-8')
train.to_csv('train_data.csv',index=False,encoding='utf-8')


# In[72]:


# loading training data csv file
train_df = pd.read_csv('train_data.csv')
train_df.head()


# In[73]:


# splitting training data into train and cross validation dataset 
train_data,cv_data=train_test_split(train_df,test_size=0.3)


# In[74]:


# separating features and labels of training dataset
x_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,1].values


# In[75]:


# saving cross validation data into csv file
cv_data.to_csv('cv_data.csv',index=False,encoding='utf-8')


# In[76]:


# fitting simple linear regression model to the training dataset
# lin_reg = LinearRegression(normalize=True)  
# lin_reg.fit( x_train, y_train)  

# fitting polynomial regression model to the training dataset
poly_reg = PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x_train)
# fit into multiple Linear regression model
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)


# In[77]:


y_pred=lin_reg2.predict(poly_reg.fit_transform(x_train))
pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})                                


# In[78]:


# loading cross validation dataset file
cv_data = pd.read_csv('cv_data.csv')
cv_data.head()


# In[79]:


# separate labels and features of cross validation dataset
x_cv=cv_data.iloc[:,:-1].values
y_cv=cv_data.iloc[:,1].values


# In[80]:


class PolynomialRegression:
    
    def fit_model(self,x_train,y_train):        
        # getting prediction values on train dataset
        y_pred_train=lin_reg2.predict(poly_reg.fit_transform(x_train))
        
        return y_pred_train
    
    def fit_model_cv(self,x_cv):
        # getting prediction values cross validation dataset 
        y_pred=lin_reg2.predict(poly_reg.fit_transform(x_cv))
        return y_pred
    
    def get_accuracy(self,y_train,y_pred):
        Accuracy = sklearn.metrics.r2_score(y_train,y_pred)*100
        return Accuracy
        
    def visualize_trainset(self,y_pred_train):
        # visualizing the training set result
        x_grid=np.arange(min(x_train),max(x_train),0.1)
        x_grid=x_grid.reshape((len(x_grid),1))
        plt.scatter(x_train,y_train,color='red')
        plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
        plt.title('predict number of bikes getting shared (Training Set)')
        plt.xlabel('Registered')
        plt.ylabel('No. of bikes ')
        plt.show()
        
    def visualize_cvset(self,y_pred_test):
        # visualizing the testing set result
        x_grid=np.arange(min(x_cv),max(x_cv),0.1)
        x_grid=x_grid.reshape((len(x_grid),1))
        plt.scatter(x_cv,y_cv,color='red')
        plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
        plt.title('predict number of bikes getting shared (Cross Validation Set)')
        plt.xlabel('Registered')
        plt.ylabel('No. of bikes')
        plt.show()
        
def main():
    # creates class object 
    obj = PolynomialRegression()
    y_pred_train = obj.fit_model(x_train,y_train)
    
    y_pred_test = obj.fit_model_cv(x_cv)
    
    acc_train = obj.get_accuracy(y_train,y_pred_train)
    print("Accuracy of train data =",acc_train)
    
    acc_test = obj.get_accuracy(y_cv,y_pred_test)
    print("Accuracy of test data =",acc_test)
     
    obj.visualize_trainset(y_pred_train)
    obj.visualize_cvset(y_pred_test)
    
if __name__ == '__main__':
    main()
    


# In[81]:


fileObject = open("train_data.pkl",'wb')
pickle.dump(poly_reg,fileObject)  
pickle.dump(lin_reg2,fileObject)   
# here we close the fileObject
fileObject.close()


# In[ ]:




