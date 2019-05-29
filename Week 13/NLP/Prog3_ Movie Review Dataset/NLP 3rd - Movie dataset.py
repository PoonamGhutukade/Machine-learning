#!/usr/bin/env python
# coding: utf-8

# # NLP 3rd - Movie dataset
#     3. Large Movie Review Dataset
#     
# We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 
# There is additional unlabeled data for use as well. Raw text and already processed bag of words formats 
# are provided
# 

# In[1]:


import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
# cleaning the text
import re


# In[2]:


# Load the dataset
# here category for +ve or -ve category
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print(documents[1])
# take list of all words in the document
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

print("Most common words:\n",all_words.most_common(15))
# print(all_words["stupid"])


# In[3]:


"""compiling feature lists of words from positive reviews and words from the negative reviews to 
hopefully see trends in specific types of words in positive or negative reviews."""
# take 3000 most common words
word_features = list(all_words.keys())[:3000]


# In[4]:


# dictionnary of 3000 words
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# In[5]:


print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))


# In[6]:


# It will find features & convert list
featuresets = [(find_features(rev), category) for (rev, category) in documents]
#now that we have our features and labels


# In[7]:


len(featuresets)


# In[9]:


# featuresets = pd.DataFrame(featuresets)
# print(featuresets.shape, type(featuresets))
# featuresets.head()
# featuresets.columns


# separate our data into training and testing sets, and use Naive Bayes classifier. 
# This is a pretty popular algorithm used in text classification

# In[21]:


# set that we'll train our classifier with
training = featuresets[:1600]
# set that we'll test against.
testing_set = featuresets[1600:]

# print(type(testing_set))
# convert list into dataframe and load into csv file 
# testing_set = pd.DataFrame(testing_set)
# testing_set.to_csv("CSV_files/Testing_file.csv")

#Saving testing file into pickle file
test_file = open("CSV_files/Testing_file.csv","wb")
pickle.dump(testing_set, test_file)
test_file.close()

# split dataset into train and cross validation
training_set = training[:1200]
cross_validation = training[1200:]

print("Training set:",len(training_set),"Cross_ validation:",  len(cross_validation),"Testing set:", len(testing_set))


# ### Build Naive Bayes Model for NLP 

# In[13]:


classifier = nltk.NaiveBayesClassifier.train(training_set)
#First we just simply are invoking the Naive Bayes classifier,
#then we go ahead and use .train() to train it all in one line.
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, cross_validation))*100)


# In[14]:


classifier.show_most_informative_features(15)


# In[15]:


#Saving Classifiers in pickle file
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# ### DecisionTreeClassifier Model

# In[18]:


#First we just simply are invoking the DecisionTreeClassifier
classifier_1 = nltk.DecisionTreeClassifier.train(training_set)


# In[20]:


#then we go ahead and use .train() to train it all in one line.
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier_1, cross_validation))*100)

