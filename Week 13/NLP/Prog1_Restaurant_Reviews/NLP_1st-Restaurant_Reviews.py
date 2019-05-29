#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing	
# 1. Using NLP predict whether the review is positive or negative for a given dataset 
# 

# In[28]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# Feature Scaling
from sklearn.preprocessing import StandardScaler
# NLTK libraries
import nltk
# cleaning the text
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn import metrics

#Classification library
from sklearn.tree import DecisionTreeClassifier

#Classification library
from sklearn.ensemble import RandomForestClassifier

# precision recall
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
import importlib.util


# In[2]:


# importing template file 
spec = importlib.util.spec_from_file_location("Util_class", "/home/admin1/PycharmProjects/Machine-Learning/Week 13/Util/util.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
# creating object of Template class
obj_util = foo.Util_class()


# In[3]:


#Load dataset
# tsv = tab seperated value, to read this file use delimiter, and to remove double quotes use quoting = 3
dataset_original = pd.read_csv ("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
dataset = dataset_original
dataset.head()


# In[4]:


print("Dataset has {} rows and {} Columns".format(dataset.shape[0],dataset.shape[1])) 


# In[5]:


# check dataset information
dataset.info()


# In[6]:


# descibe the dataset
dataset.describe().T


# In[7]:


# handling missing data if nessesary
dataset.isnull().sum()


# In[8]:


# check for duplicate values
dataset.duplicated().sum()


# In[9]:


# dataset = dataset.drop_duplicates()


# In[10]:


# Download nltk stopwords
nltk.download("stopwords")


# In[11]:


dataset.columns


# In[12]:


# cleaning the text
corpus = []
for i in range(dataset.shape[0]):
    # step1 -> Only taking letters
    # Here we only taking latter, others are removed
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
#     print(review)

    # step2 -> Convert all words into lower case
    review = review.lower()
#     print(review)
    # split to array(default delimiter is " ") 
    # Step3 -> take sentence into words
    #  Tokenization, involves splitting sentences and words from the body of the text.
    review = review.split()
#     print(review)

    # Step4 -> Remove Unnecessary words using stopwords 
    # Step5 -> Take rootwords using stem() func by importing PorterStemmer

    ps = PorterStemmer()
    # here we take set() function bcoz it goes through all the word faster to all words in list
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     print(review)

    # Step6 -> join all words
    review = ' '.join(review)
    
    # Step7 -> Append whole list
    corpus.append(review)
    


# In[13]:


review


# In[14]:


# Step8 -> Creating the Bag of words model
# taking unique words and take one column fr one word then add all cols into table. When cols created for 
#each word it is each one is independent variable, & it allow us for classification model through the process 
#of tokenizer

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# In[15]:


x.shape, y.shape


# In[24]:


# import os
# create directory to store csv files
# os.mkdir("CSV_files")


# ### NLP by using - Naive_Bayes

# In[29]:


class Naive_Bayes():
    
    def create_module(self,x_train,y_train):
        # Fitting Naive Bayes to the Training set
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        return classifier
        
def main():
    #class obj created
    obj  = Naive_Bayes()
    
    #split dataset into train, test and cross validation also save csv files
    x_train, x_cv,  y_train, y_cv = obj_util.splitdata(x, y, "CSV_files/Naive_Bayes_test.csv", 0.20,0.20)

#     # Feature Scalling for train and test data 
#     x_train = obj_util.feature_Scaling(x_train)
#     x_test = obj_util.feature_Scaling(x_test)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_cv = sc.transform(x_cv)

    classifier = obj.create_module(x_train,y_train)
#     cl = nltk.NaiveBayesClassifier.train(x_train)
#     print(cl.show_most_informative_features(10))
#     print("\nModule created")
#     print("classifier object",classifier)

    y_pre = obj_util.y_prediction(x_cv, classifier)
#     print("\n\n y_prediction:",y_pre)
#     print(y_pre.shape)
    
    cm = obj_util.confusion_matrix(y_cv,y_pre)
    print("\n\nConfusion Matrix:\n",cm)
    
    accuracy_score,average_precision,auc, f1_score_acc = obj_util.accuracy(y_pre,y_cv)
    print('\n\nAverage accuracy_score:' , accuracy_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Average Roc-AUC: %.3f' % auc)
    print('Accuracy by F1-score:',f1_score_acc)

    obj_util.create_piklefile(classifier, sc, "Naive_Bayes.pkl")
    print("\nModel created with pickle file")
    
    print("-------------------------CROSS Validation-------------------------------------")
     # predict y
    y_pre = obj_util.y_prediction(x_cv, classifier)
    # Calculate accuracy
    accuracy = average_precision_score(y_cv, y_pre)* 100
    print("\n Accuracy: average_precision_score :", accuracy)

    print("\n",classification_report(y_cv, y_pre))

    y_pred_prob = classifier.predict_proba(x_cv)[:,1]

    print("\n ROC curve \n")
    fpr, tpr, thresholds =  metrics.roc_curve(y_cv, y_pred_prob)
    # create plot
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.legend(loc="lower right")

if __name__ == '__main__':
    main()


# ### NLP by using - Decision Tree Classification 

# In[ ]:





# In[30]:


class DecisionTreeClassification():
    
    def create_module(self,x_train,y_train):
        # Fitting Decision Tree Classification to the Training set
        classifier = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
        classifier.fit(x_train,y_train)
        return classifier
        
def main():
    #class obj created
    obj  = DecisionTreeClassification()
    
    #split dataset into train, test and cross validation also save csv files
    x_train, x_cv,  y_train, y_cv = obj_util.splitdata(x, y, "CSV_files/DecisionTree_Test.csv", 0.30,0.20)
    
    # feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_cv = sc.transform(x_cv)

    # fit model 
    classifier = obj.create_module(x_train,y_train)

    # predict y
    y_pre = obj_util.y_prediction(x_train, classifier)
    
    cm = obj_util.confusion_matrix(y_train,y_pre)
    print("\n\nConfusion Matrix:\n",cm)
    
    accuracy_score,average_precision,auc, f1_score_acc = obj_util.accuracy(y_pre,y_train)
    
    print('\n\nAverage accuracy_score:' , accuracy_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Average Roc-AUC: %.3f' % auc)
    print('Accuracy by F1-score:',f1_score_acc)

    obj_util.create_piklefile(classifier, sc, "DecisionTreeClassification.pkl")
    print("\nModel created with pickle file")
    
    print("-------------------------CROSS Validation-------------------------------------")
     # predict y
    y_pre = obj_util.y_prediction(x_cv, classifier)
    
    accuracy = average_precision_score(y_cv, y_pre)* 100
    print("\n Accuracy: average_precision_score :", accuracy)


    print("\n",classification_report(y_cv, y_pre))

    y_pred_prob = classifier.predict_proba(x_cv)[:,1]

    print("\n ROC curve \n")
    fpr, tpr, thresholds = roc_curve(y_cv, y_pred_prob)
    # create plot
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.legend(loc="lower right")
    
if __name__ == '__main__':
    main()


# ### NLP by using - Random_forest Classification 

# In[ ]:





# In[31]:


class Random_forest():
    
    def create_module(self,x_train,y_train):
        # Fitting Random_forest Classification to the Training set
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(x_train,y_train)
        return classifier
        
def main():
    #class obj created
    obj  = Random_forest()
    
 #split dataset into train, test and cross validation also save csv files
    x_train, x_cv,  y_train, y_cv = obj_util.splitdata(x, y, "CSV_files/Random_forest_Test.csv", 0.30,0.20)
    
    # feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_cv = sc.transform(x_cv)

    # fit model 
    classifier = obj.create_module(x_train,y_train)

    # predict y
    y_pre = obj_util.y_prediction(x_cv, classifier)
    
    cm = obj_util.confusion_matrix(y_cv,y_pre)
    print("\n\nConfusion Matrix:\n",cm)
    
    accuracy_score,average_precision,auc, f1_score_acc = obj_util.accuracy(y_pre,y_cv)
    
    print('\n\nAverage accuracy_score:' , accuracy_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Average Roc-AUC: %.3f' % auc)
    print('Accuracy by F1-score:',f1_score_acc)

    obj_util.create_piklefile(classifier,sc, "Random_forest.pkl")
    print("\nModel created with pickle file")
    
    print("-------------------------CROSS Validation-------------------------------------")
     # predict y
    y_pre = obj_util.y_prediction(x_cv, classifier)
    
    accuracy = average_precision_score(y_cv, y_pre)* 100
    print("\n Accuracy: average_precision_score :", accuracy)


    print("\n",classification_report(y_cv, y_pre))

    y_pred_prob = classifier.predict_proba(x_cv)[:,1]

    print("\n ROC curve \n")
    fpr, tpr, thresholds = roc_curve(y_cv, y_pred_prob)
    # create plot
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.legend(loc="lower right")

if __name__ == '__main__':
    main()


# In[ ]:




