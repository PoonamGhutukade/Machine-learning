#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Classification
	Random Forest Classification
2. The data contains lists of octamers (8 amino acids) and a flag (-1 or 1) depending on whether HIV-1 
protease will cleave in the central position (between amino acids 4 and 5). Build a machine learning  model 
for the dataset, please refer document inside zip file for additional information
"""


# In[2]:


#import libraries
import pandas as pd
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt  
#library for feature scaling
from sklearn.preprocessing import StandardScaler
# #Classification library
from sklearn.ensemble import RandomForestClassifier

import pickle 
import os, sys
import csv
# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import importlib.util


# In[3]:


# importing template file 
spec = importlib.util.spec_from_file_location("Util_class", "/home/admin1/PycharmProjects/Machine-Learning/Week10/Util/util.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
obj_util = foo.Util_class()


# In[4]:


# load dataset
dataset_original = pd.read_csv ("1625Data.txt", delimiter = ",",names=["Peptides", "Result"])
dataset = dataset_original
dataset.head()


# In[5]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[6]:


# check dataset information
dataset.info()


# In[7]:


# descibe the dataset
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


#check for NAN values
dataset.isna().sum()


# In[10]:


# check for duplicate values
dataset.duplicated().sum()


# In[11]:


# check skewness for target variable
sb.distplot(dataset['Result'])
print ("Skewness of y is {}".format(dataset['Result'].skew()))


# In[12]:


def remove_skew_square():
    print("\nSkewness for Target")
    dataset['Result'] = (np.sqrt(dataset['Result']))
    print("Mean: ",dataset['Result'].mean(),"Median: ", dataset['Result'].median(), 'Skewness is :', dataset['Result'].skew())

    print("Draw histogram")
    plt.hist(dataset['Result'])
    plt.show()
    
# remove_skew_square()
# Here if we remove skewness result gives large amount of NAN values


# In[13]:


dataset.isna().sum()


# In[14]:


# Seperate all amino acids
peptides = np.array([[dataset["Peptides"][i][j] for i in range(dataset.shape[0])] for j in range(8)])
peptides.shape


# In[15]:


# Store the seperated amino acids into a dataframe
dataset2 = pd.DataFrame(peptides.T, columns=list('ABCDEFGH'))
dataset2.shape


# In[16]:


dataset2.head()


# In[17]:


# assign 2nd dataset to 1st one
dataset = dataset.assign(**dataset2)
#OR
# dataset = pd.concat([dataset,dataset2])
dataset.head()


# In[18]:


# drop unwanted column
dataset = dataset.drop(['Peptides'], axis=1)


# In[19]:


dataset = dataset[['A','B','C','D','E','F','G','H','Result']]
dataset.head()


# In[20]:


print("Dataset shape",dataset.shape)
dataset.head()


# In[21]:


# create directory to store csv files
os.mkdir("CSV_files")


# In[22]:


#split dataset into train, test and cross validation also save csv files
obj_util.splitdata(dataset, 0.30, 0.40,"CSV_files" )


# In[23]:


# load train dataset
train_dataset = pd.read_csv ("CSV_files/train_file.csv")
print("Train Dataset has {} rows and {} Columns".format(train_dataset.shape[0],train_dataset.shape[1])) 
# load dataset for Cross Validation
CV_dataset = pd.read_csv ("CSV_files/CValidation_file.csv")
print("Cross validation Dataset has {} rows and {} Columns".format(CV_dataset.shape[0],CV_dataset.shape[1])) 


# In[24]:


#data Preprocessing
train_dataset.info()


# In[25]:


train_dataset.head()


# In[26]:


# seperate fetures and label

x_train = train_dataset.loc[:, train_dataset.columns != 'Result'].values
y_train = train_dataset.loc[:,train_dataset.columns == 'Result'].values

# convert ndarray to dataframe
df1 =  pd.DataFrame(x_train)

print("x_train :",x_train.shape,"& y_train:",y_train.shape)

#for cross validation
x_crossval = CV_dataset.loc[:, CV_dataset.columns != 'Result'].values
y_crossval = CV_dataset.loc[:,CV_dataset.columns == 'Result'].values

# convert ndarray to dataframe
df2 =  pd.DataFrame(x_crossval)
print("x_cv :",x_crossval.shape,"& y_cv:",y_crossval.shape)


# In[27]:


df2.shape


# In[28]:


# Handle categorical data
x_train_dataset = obj_util.Categorical_data(df1)


# In[29]:


# Handle categorical data for cross validation dataset
cv_dataset = obj_util.Categorical_data(df2)


# In[31]:


x_train_dataset.shape, cv_dataset.shape


# In[32]:


#feature scalling (here data will be converted into float)
def feature_scalling(x_train,x_crossval):
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    
    sc_x_cv = StandardScaler()
    x_crossval = sc_x.fit_transform(x_crossval)
    
    return sc_x, x_train,sc_x_cv, x_crossval
    
sc_x, x_train,sc_x_cv, x_crossval = feature_scalling(x_train_dataset,cv_dataset)


# In[33]:


print(len(x_train), len(x_crossval))


# In[34]:


class Random_forest():
    
    # create random forest Model
    def create_module(self,x_train,y_train):
        # fitting KNN Classification to the training set
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(x_train,y_train)
        return classifier
    

def main():
    #class obj created
    obj  = Random_forest()

     # create random forest Model
    classifier = obj.create_module(x_train,y_train)
    print("\nModule created")
    print("classifier object",classifier)

    # y prediction
    y_pre = obj_util.y_prediction(x_train, classifier)
 
    
    # calculate accuracy
    accuracy_score,average_precision,auc=obj_util.accuracy(y_pre,y_train)
    print('\n\nAverage accuracy_score:' , accuracy_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Average Roc-AUC: %.3f' % auc)

    # show confusion matrix
    cm = obj_util.confusion_matrix(y_train,y_pre)
    print("\n\nConfusion Matrix:\n",cm)
    
    # data visualisation
#     obj_util.visualization(x_train,y_train, classifier, "Random_forest Classification(Training set)", "Age", "Estimate Salary")
    
    # create pickle file
    obj_util.create_piklefile(classifier,'Random_forest1.pkl' )
    print("\nPikle file created")


if __name__ == '__main__':
    main()


# In[35]:


# cross validation        
def Cross_validation():
    # load pickle file 
    file1 = open('Random_forest1.pkl', 'rb')
    classifier1 = pickle.load(file1)

    # y_prediction ( cross validation) 
    y_predicted1 = obj_util.y_prediction(x_crossval, classifier1)
#     print("\n\n y_prediction:",y_predicted1)
    
#     print(y_crossval.shape, y_predicted1.shape)
    # calculate accuracy
    accuracy_score,average_precision,auc=obj_util.accuracy(y_predicted1, y_crossval)
    print('\n\nAverage accuracy_score:' , accuracy_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Average Roc-AUC: %.3f' % auc)

    # show confusion matrix
    cm = obj_util.confusion_matrix(y_crossval, y_predicted1)
    print("\n\nConfusion Matrix:\n",cm)
#     print("\n\nConfusion Matrix:\n",metrics.confusion_matrix(y_crossval, y_predicted1))
    
#     obj_util.visualization(x_crossval, y_crossval, classifier1, "Random_forest(Cross_validation set)", "Age", "Estimate Salary")
    
Cross_validation()


# In[ ]:





# In[ ]:




