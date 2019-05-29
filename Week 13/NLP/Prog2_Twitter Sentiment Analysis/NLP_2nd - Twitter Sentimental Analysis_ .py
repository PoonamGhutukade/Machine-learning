
#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing	
# 6. Twitter Sentiment Analysis
# 
# Hate Speech in the form of racism and sexism has become a nuisance on twitter and it is important to segregate these sort of tweets from the rest. We provide Twitter data that has both normal and hate tweets. Your task as a Data Scientist is to identify the tweets which are hate tweets and which are not.
# 

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# Importing HTMLParser
from html.parser import HTMLParser
# Speling correlation with textblob
from textblob import TextBlob
# Importing stop words from NLTK coupus and word tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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


# Loading Data
train_df = pd.read_csv ("twitter-sentiment-analysis/train_E6oV3lV.csv")
print("Train Dataset has {} rows and {} Columns".format(train_df.shape[0],train_df.shape[1])) 
train_df.head(15)


# In[4]:


test_df = pd.read_csv ("twitter-sentiment-analysis/test_tweets_anuFYb8.csv")
print("Test Dataset has {} rows and {} Columns".format(test_df.shape[0],test_df.shape[1])) 
test_df.head(15)


# In[5]:


# check dataset information
train_df.info()


# In[6]:


# check dataset information
test_df.info()


# We can see in above tweet column in both data sets Training & Testing tweets are unstructured, for better 
# analysis we first need to structure the tweets, remove the unwanted words, replace the misspelled words 
# with the correct ones, replace the abriviation with full words

# In[7]:


# Merging both the data sets as tweets in both the data set is unstructured
combine_df = train_df.append(test_df, ignore_index = True, sort = False)
combine_df.head(15)


# In[8]:


combine_df.info()


# In[9]:


# descibe the dataset
combine_df.describe().T


# In[10]:


# handling missing data if nessesary
combine_df.isnull().sum()


# In[11]:


# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values= 'Nan',axis=0)
# combine_df['label'] = combine_df['label'].reshape(-1, 1)
# imputer = imputer.fit(combine_df['label'])
# label = imputer.transform(label)

# replace  NaN values with mean value
combine_df.replace(np.NaN,combine_df.mean(),inplace=True)


# In[12]:


# handling missing data if nessesary
combine_df.isnull().sum()


# In[13]:


# check for duplicate values
combine_df.duplicated().sum()


# ## Data Preprocessing & cleaning

# ### Step 1 : Converting html entities

# In[14]:


"""Step 1 : Converting html entities i.e. (&lt; &gt; &amp;)
( "&lt;" is converted to “<” and "&amp;" is converted to “&”)

example - 
$str = "Albert Einstein said: 'E=MC²'";
echo htmlentities($str, ENT_COMPAT); // Will only convert double quotes
echo "<br>";

output->
Albert Einstein said: 'E=MC&sup2;'<br>
"""
# Importing HTMLParser
html_parser = HTMLParser()


# In[15]:


# Created a new columns i.e. clean_tweet contains the same tweets but cleaned version
combine_df['clean_tweet'] = combine_df['tweet'].apply(lambda x: html_parser.unescape(x))
combine_df.head()


# ### Step 2 : Removing "@user" from all the tweets

# In[16]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        # here we substract all @ signs
        input_txt = re.sub(i, '', input_txt)
    return input_txt


# In[17]:


# remove twitter handles (@user)
combine_df['clean_tweet'] = np.vectorize(remove_pattern)(combine_df['clean_tweet'], "@[\w]*")
combine_df.head(10)


# ### Step 3 : Changing all the tweets into lowercase¶
# 

# In[18]:


combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: x.lower())
combine_df.head()


# ### Step 4 : Apostrophe Lookup

# In[19]:


# Apostrophe Dictionary
apostrophe_dict = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
apostrophe_dict


# In[20]:


def lookup_dict(text, dictionary):
    # 1st we have to split the sentences
    for word in text.split():
        if word.lower() in dictionary:
            if word.lower() in text.split():
                text = text.replace(word, dictionary[word.lower()])
    return text


# In[21]:


combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: lookup_dict(x,apostrophe_dict))
combine_df.head(10)


# ### Step 5 : Short Word Lookup
# 

# In[22]:


short_word_dict = {
"121": "one to one",
"a/s/l": "age, sex, location",
"adn": "any day now",
"afaik": "as far as I know",
"afk": "away from keyboard",
"aight": "alright",
"alol": "actually laughing out loud",
"b4": "before",
"b4n": "bye for now",
"bak": "back at the keyboard",
"bf": "boyfriend",
"bff": "best friends forever",
"bfn": "bye for now",
"bg": "big grin",
"bta": "but then again",
"btw": "by the way",
"cid": "crying in disgrace",
"cnp": "continued in my next post",
"cp": "chat post",
"cu": "see you",
"cul": "see you later",
"cul8r": "see you later",
"cya": "bye",
"cyo": "see you online",
"dbau": "doing business as usual",
"fud": "fear, uncertainty, and doubt",
"fwiw": "for what it's worth",
"fyi": "for your information",
"g": "grin",
"g2g": "got to go",
"ga": "go ahead",
"gal": "get a life",
"gf": "girlfriend",
"gfn": "gone for now",
"gmbo": "giggling my butt off",
"gmta": "great minds think alike",
"h8": "hate",
"hagn": "have a good night",
"hdop": "help delete online predators",
"hhis": "hanging head in shame",
"iac": "in any case",
"ianal": "I am not a lawyer",
"ic": "I see",
"idk": "I don't know",
"imao": "in my arrogant opinion",
"imnsho": "in my not so humble opinion",
"imo": "in my opinion",
"iow": "in other words",
"ipn": "I’m posting naked",
"irl": "in real life",
"jk": "just kidding",
"l8r": "later",
"ld": "later, dude",
"ldr": "long distance relationship",
"llta": "lots and lots of thunderous applause",
"lmao": "laugh my ass off",
"lmirl": "let's meet in real life",
"lol": "laugh out loud",
"ltr": "longterm relationship",
"lulab": "love you like a brother",
"lulas": "love you like a sister",
"luv": "love",
"m/f": "male or female",
"m8": "mate",
"milf": "mother I would like to fuck",
"oll": "online love",
"omg": "oh my god",
"otoh": "on the other hand",
"pir": "parent in room",
"ppl": "people",
"r": "are",
"rofl": "roll on the floor laughing",
"rpg": "role playing games",
"ru": "are you",
"shid": "slaps head in disgust",
"somy": "sick of me yet",
"sot": "short of time",
"thanx": "thanks",
"thx": "thanks",
"ttyl": "talk to you later",
"u": "you",
"ur": "you are",
"uw": "you’re welcome",
"wb": "welcome back",
"wfm": "works for me",
"wibni": "wouldn't it be nice if",
"wtf": "what the fuck",
"wtg": "way to go",
"wtgp": "want to go private",
"ym": "young man",
"gr8": "great"
}


# In[23]:


combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: lookup_dict(x,short_word_dict))
combine_df.head(10)


# ### Step 6 : Emoticon Lookup

# In[24]:


emoticon_dict = {
":)": "happy",
":‑)": "happy",
":-]": "happy",
":-3": "happy",
":->": "happy",
"8-)": "happy",
":-}": "happy",
":o)": "happy",
":c)": "happy",
":^)": "happy",
"=]": "happy",
"=)": "happy",
"<3": "happy",
":-(": "sad",
":(": "sad",
":c": "sad",
":<": "sad",
":[": "sad",
">:[": "sad",
":{": "sad",
">:(": "sad",
":-c": "sad",
":-< ": "sad",
":-[": "sad",
":-||": "sad"
}
# emoticon_dict


# In[25]:


combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: lookup_dict(x,emoticon_dict))
combine_df.head()


# ### Step 7 : ReplacingPunctuations with space
# 

# In[26]:


combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
combine_df.head(10)


# ### Step 8 : Replacing Special Characters, Numbers (integers)  with space
# 

# In[27]:


combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))
combine_df.head(10)


# ### Step 9 :Removing words whom length is 1¶
# 
# 

# In[28]:


combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
combine_df['clean_tweet'][0:5]


# ### Step 10 : Spelling Correction - With TextBlob Library

# In[30]:


# Not cleaning the just showing the spelling check as its take lot of time to process all these tweets
## Shown sample how its must done
text = combine_df['clean_tweet'][0:10].apply(lambda x: str(TextBlob(x).correct()))
text


# ### Cleaning

# #### 1.word_tokenize

# In[32]:


# Creating token for the clean tweets
combine_df['tweet_token'] = combine_df['clean_tweet'].apply(lambda x: word_tokenize(x))

## Fully formated tweets & there tokens
combine_df.head()


# #### 2.stopwords

# In[33]:


# Importing stop words from NLTK corpus for english language
stop_words = set(stopwords.words('english'))
# stop_words


# In[34]:


# Created new columns of tokens - where stop words are being removed
combine_df['tweet_token_filtered'] = combine_df['tweet_token'].apply(lambda x: [word for word in x if not word in stop_words])

## Tokens columns with stop words and without stop words
combine_df[['tweet_token', 'tweet_token_filtered']].head()


# #### 3.Stemming & Lemmatization
# We will create 2 new columns
#     
#     1.One For Stemming
#     2.Second For Lemmatization
#     
#     The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.
# 
# Stemming - Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. 

# In[35]:


#It take rootwords using stem() func by importing PorterStemmer
stemming = PorterStemmer()


# In[36]:


# Created one more columns tweet_stemmed it shows tweets' stemmed version
combine_df['tweet_stemmed'] = combine_df['tweet_token_filtered'].apply(lambda x: ' '.join([stemming.stem(i) for i in x]))
combine_df['tweet_stemmed'].head(10)


# #### 4. Lemmatization -
#     Lemmatization is the process of converting a word to its base form.¶
# 

# In[ ]:


# Importing library for stemming
from nltk.stem import PorterStemmer
# Importing library for lemmatizing
from nltk.stem.wordnet import WordNetLemmatizer
#visualizing all the words in column "tweet_stemmed" in our data using the wordcloud plot.
from wordcloud import WordCloud
# Importing library
from sklearn.feature_extraction.text import CountVectorizer


# In[37]:



lemmatizing = WordNetLemmatizer()


# In[38]:


# Created one more columns tweet_lemmatized it shows tweets' lemmatized version
combine_df['tweet_lemmatized'] = combine_df['tweet_token_filtered'].apply(lambda x: ' '.join([lemmatizing.lemmatize(i) for i in x]))
combine_df['tweet_lemmatized'].head()


# In[39]:


# Our final dataframe - Fully formatted, Processed, Noise less, Cleaned, ready to analyse
## for further analysis we consider 2 columns i.e. "tweet_stemmed" & "tweet_lematized"
### We are using 2 columns to see which of them give us better score.
combine_df.head(10)


# A - Will see the most commonly used words for both the columns i.e. "tweet_stemmed" & "tweet_lematized"
# 

# In[ ]:





# In[40]:


#visualizing all the words in column "tweet_stemmed" in our data using the wordcloud plot.
all_words = ' '.join([text for text in combine_df['tweet_stemmed']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most Common words in column Tweet Stemmed")
plt.show()


# In[41]:


#visualizing all the words in column "tweet_stemmed" in our data using the wordcloud plot.
all_words = ' '.join([text for text in combine_df['tweet_lemmatized']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most Common words in column Tweet Stemmed")
plt.show()


# In[42]:


#Visualizing all the normal or non racist/sexist words in column "tweet_stemmed" in our data using the wordcloud plot.
normal_words =' '.join([text for text in combine_df['tweet_lemmatized'][combine_df['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most non racist/sexist words in column Tweet Stemmed")
plt.show()
"""In above plots we can see most of the words are positive or neutral. With happy and love being the 
most frequent ones.
"""


# In[43]:


#Visualizing all the negative or racist/sexist words in column "tweet_stemmed" in our data using the wordcloud plot.
negative_words =' '.join([text for text in combine_df['tweet_stemmed'][combine_df['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most racist/sexist words in column Tweet Stemmed")
plt.show()

"""In above plots we can see most of the words are negative or racist/sexist. 
With racist, hate and black being the most frequent ones.
"""


# #### 5. A - Bag-of-Words Features

# In[ ]:





# In[44]:



bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow_vectorizer


# In[45]:


# bag-of-words feature matrix - For columns "combine_df['tweet_stemmed']"
bow_stem = bow_vectorizer.fit_transform(combine_df['tweet_stemmed'])
# print(bow_stem)
# bag-of-words feature matrix - For column - combine_df['tweet_lemmatized']
bow_lemm = bow_vectorizer.fit_transform(combine_df['tweet_lemmatized'])
bow_lemm


# In[46]:


# Importing library
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf_vectorizer


# In[47]:


# TF-IDF feature matrix - For columns "combine_df['tweet_stemmed']"
tfidf_stem = tfidf_vectorizer.fit_transform(combine_df['tweet_stemmed'])
# tfidf_stem
# TF-IDF feature matrix - For columns "combine_df['tweet_lemmatized']"
tfidf_lemm = tfidf_vectorizer.fit_transform(combine_df['tweet_lemmatized'])
tfidf_lemm


# In[48]:


combine_df.head()


# ### Model Building 

# In[54]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# A.1 For columns "combine_df['tweet_stemmed']"
train_bow = bow_stem[:31962,:]
test_bow = bow_stem[31962:,:]
#use getnnz() or shape[0]
print(train_bow.shape)
# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_df['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

A1 = f1_score(yvalid, prediction_int)*100 # calculating f1 score
print(A1)


# In[55]:


# A.2 For columns "combine_df['tweet_lemmatized']"
train_bow = bow_lemm[:31962,:]
test_bow = bow_lemm[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_df['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

A2 = f1_score(yvalid, prediction_int)*100 # calculating f1 score
print(A2)


# In[56]:


# B.1 For columns "combine_df['tweet_stemmed']"
train_tfidf = tfidf_stem[:31962,:]
test_tfidf = tfidf_stem[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

B1 = f1_score(yvalid, prediction_int)*100 # calculating f1 score
print(B1)


# In[57]:


# B.2 For columns "combine_df['tweet_lemmatized']"
train_tfidf = tfidf_lemm[:31962,:]
test_tfidf = tfidf_lemm[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

B2 = f1_score(yvalid, prediction_int)*100 # calculating f1 score
print(B2)


# In[58]:


print("F1 - Score Chart")
print("* F1-Score - Model using Bag-of-Words features")
print("   F1-Score = ",A1," - For column tweets are stemmed")
print("   F1-Score = ",A2," - For column tweets are Lemmatized")
print("* F1-Score - Model using TF-IDF features")
print("   F1-Score = ",B1," - For column tweets are stemmed")
print("   F1-Score = ",B2," - For column tweets are Lemmatized")


# In[ ]:




