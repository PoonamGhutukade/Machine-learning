#!/usr/bin/env python
# coding: utf-8

# # Association Rule Learning
# 
# Apriori
# 1. Generate association rules for dataset given in the url
# 
# Apriori algorithm to find out which items are commonly sold together, so that store owner can take action to place the related items together or advertise them together in order to have increased profit.

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

# association rule algorithm file
from apyori import apriori


# In[2]:


# Load the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


# In[3]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[4]:


# check dataset information
dataset.info()


# In[5]:


# descibe the dataset
dataset.describe()


# In[6]:


# check for duplicate values
dataset.duplicated().sum()


# In[7]:


# Apriory expecting list of list as an input so we required two loops
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0, dataset.shape[1])])


# ### There are three major components of Apriori algorithm: 
#     
# 1.Support -> Support is the basic probability of an event to occur.
# 
#     -Support refers to the default popularity of an item
#     Support(B) = (Transactions containing (B))/(Total Transactions)  
# 
# 2.Confidence -> The confidence of an event is the conditional probability of the occurrence;
#     
#     -Confidence refers to the likelihood that an item B is also bought if item A is bought.
#     Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)  
# 
# 3.Lift -> This is the ratio of confidence to expected confidence
# 
#     -Lift(A -> B) refers to the increase in the ratio of sale of B when A is sold
#     Lift(A→B) = (Confidence (A→B))/(Support (B))  
# 

# In[8]:


# Training Apriori on the dataset
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# In[9]:


# Visualising the results
association_rules = list(rules)


# In[10]:


print(len(association_rules)) 


# In[11]:


print(association_rules[0])  


# In[12]:


for item in association_rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("________________________________________________")


# In[ ]:




