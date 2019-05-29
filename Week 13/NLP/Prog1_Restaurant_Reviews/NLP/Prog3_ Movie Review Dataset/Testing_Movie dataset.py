#!/usr/bin/env python
# coding: utf-8

# In[22]:


import nltk
import random
from nltk.corpus import movie_reviews
import pickle
import pandas as pd


# ### Final Tasting on Dataset

# In[18]:


# Load testing file 
test_file = open("CSV_files/Testing_file.csv","rb")
testing_set = pickle.load(test_file)
test_file.close()


# In[19]:


# Load naivebayes classification object
classifier_f = open("naivebayes.pickle", "rb")
classifier1 = pickle.load(classifier_f)
classifier_f.close()


# In[21]:


# calculate the accuracy
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier1, testing_set))*100)


# In[ ]:





# In[ ]:





# In[ ]:



# If I tryied to load test dataset from csv file it gives value error to unpack the values


"""ValueError                                Traceback (most recent call last)
<ipython-input-14-c7ab7b5551ac> in <module>
----> 1 print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier1, testing_set))*100)

~/anaconda3/envs/my_env/lib/python3.7/site-packages/nltk/classify/util.py in accuracy(classifier, gold)
     90 
     91 def accuracy(classifier, gold):
---> 92     results = classifier.classify_many([fs for (fs, l) in gold])
     93     correct = [l == r for ((fs, l), r) in zip(gold, results)]
     94     if correct:

~/anaconda3/envs/my_env/lib/python3.7/site-packages/nltk/classify/util.py in <listcomp>(.0)
     90 
     91 def accuracy(classifier, gold):
---> 92     results = classifier.classify_many([fs for (fs, l) in gold])
     93     correct = [l == r for ((fs, l), r) in zip(gold, results)]
     94     if correct:

ValueError: too many values to unpack (expected 2)"""

