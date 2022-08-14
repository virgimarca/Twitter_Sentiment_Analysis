#!/usr/bin/env python
# coding: utf-8

# # DSL Project
# ## Virginia Marcante, Data Science Lab, Politecnico di Torino, A.Y. 2021/2022
# ### matricule: 296312
# In this homework we will apply dataset.

# In[1]:


#LIBRARIES
import xmltodict
import csv
import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt

#Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

#NLTK
import nltk
from nltk import LancasterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.downloader.download('vader_lexicon')
nltk.download('punkt')
nltk.downloader.download('wordnet')
porter=PorterStemmer()
lancaster = LancasterStemmer()
stopwords_nltk = stopwords.words('english')
snowball_stemmer = SnowballStemmer('english')
sid = SentimentIntensityAnalyzer()

from textblob import TextBlob


# # Loading the dataset
# We start by loading the dataset as a dataframe using pandas.

# In[2]:


df_path_training = "C:\Jupyter\DSLProject\DSL2122_january_dataset\DSL2122_january_dataset\development.csv" 
df_path_test = "C:\Jupyter\DSLProject\DSL2122_january_dataset\DSL2122_january_dataset\evaluation.csv" 

col_list =  ['sentiment', 'text;;;;;;;;;']
data_train= pd.read_csv(df_path_training,sep=',', usecols=col_list, quotechar = '"', on_bad_lines='skip')
data_train= data_train.rename({'text;;;;;;;;;': 'text'}, axis=1)
data_train= data_train.loc[(data_train['sentiment'] == '0') | (data_train['sentiment'] == '1'), :]
col_list =  ['ids', 'text']
data_test = pd.read_csv(df_path_test,sep=',', usecols=col_list, quotechar = '"')


# Ratio test/(test+training)

# In[ ]:


data_train.shape[0]/(data_train.shape[0]+data_test.shape[0])*100


# In[4]:


# Histogram of tweets length
fig, ax = plt.subplots(figsize=(6,4))
ax.hist([len(x) for x in data_train['text'].values], bins = 40)
ax.set_xlabel('Length of tweets (characters)')
ax.set_ylabel('Frequency')
plt.show()
fig.savefig("hist_len_tweet.pdf")


# In[5]:


len([x for x in data_train['text'].values if len(x)>140])


# In[6]:


data_train['text'] = data_train['text'].apply(lambda x: re.sub(r'(\;)','',x))


# In[7]:


# Histogram of cleaned tweets length 
fig, ax = plt.subplots(figsize=(6,4))
ax.hist([len(x) for x in data_train['text'].values], bins = 40)
ax.set_xlabel('Length of tweets (characters)')
ax.set_ylabel('Frequency')
plt.show()
fig.savefig("hist_len_tweet_cleaned.pdf")


# In[8]:


len([x for x in data_train['text'].values if len(x)>140])


# In[9]:


vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
data_train.columns, data_test.columns


# # Preprocessing 
# The data must be cleaned before we start.
# Removal of twitter handles and websites:

# In[10]:


data_train['text'] = data_train['text'].apply(lambda x: re.sub(r'@[\w]+', '', x))
data_test['text'] = data_test['text'].apply(lambda x: re.sub(r'@[\w]+', '', x))


# In[11]:


data_train['text'] = data_train['text'].apply(lambda x: re.sub(r'http\S+', '', x))
data_test['text'] = data_test['text'].apply(lambda x: re.sub(r'http\S+', '', x))


# Computation of number of emoticon, exclamation marks, question marks, stop marks, elongated words:

# In[12]:


#Emoticons
data_train['Emoticons'] = data_train['text'].apply(lambda x: len(re.findall(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', x)))
data_test['Emoticons'] = data_test['text'].apply(lambda x: len(re.findall(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', x)))

#Exclamation Marks
data_train['ExclamationMarks'] = data_train['text'].apply(lambda x: len(re.findall(r"(\!)\1+", x)))
data_test['ExclamationMarks'] = data_test['text'].apply(lambda x: len(re.findall(r"(\!)\1+", x)))

#Question Marks
data_train['QuestionMarks'] = data_train['text'].apply(lambda x: len(re.findall(r"(\?)\1+", x)))
data_test['QuestionMarks'] = data_test['text'].apply(lambda x: len(re.findall(r"(\?)\1+", x)))

#Stop Marks
data_train['StopMarks'] = data_train['text'].apply(lambda x: len(re.findall(r"(\.)\1+", x)))
data_test['StopMarks'] = data_test['text'].apply(lambda x: len(re.findall(r"(\.)\1+", x)))

#Count Elongated
data_train['countElongated'] = data_train['text'].apply(lambda x: len([word for word in x.split() if re.compile(r"(.)\1{2}").search(word)]))
data_test['countElongated']=data_test['text'].apply(lambda x: len([word for word in x.split() if re.compile(r"(.)\1{2}").search(word)]))


# In[ ]:


Computation of text polarity:


# In[15]:


#Compound polarity
data_train['compound'] = data_train['text'].apply(sid.polarity_scores).apply(lambda x : x['compound'])
data_test['compound'] = data_test['text'].apply(sid.polarity_scores).apply(lambda x : x['compound'])
#Polarity Blob
data_train['polarity_blob'] = data_train['text'].apply(lambda x: TextBlob(x).polarity)
data_test['polarity_blob'] = data_test['text'].apply(lambda x: TextBlob(x).polarity)


# In[16]:


#posivitypolarity
data_train['pospolarity'] = data_train['text'].apply(sid.polarity_scores).apply(lambda x : x['pos'])
data_test['pospolarity'] = data_test['text'].apply(sid.polarity_scores).apply(lambda x : x['pos'])
#negativepolarity
data_train['negpolarity'] = data_train['text'].apply(sid.polarity_scores).apply(lambda x : x['neg'])
data_test['negpolarity'] = data_test['text'].apply(sid.polarity_scores).apply(lambda x : x['neg'])


# In[ ]:


Removal of special characters


# In[ ]:


data_train['text'] = data_train['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
data_test['text'] = data_test['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x)


# In[ ]:


#Stemming   
def Stemmer(sentence, stemmer):
        token_words=word_tokenize(sentence)
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(stemmer.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)
    
data_train['text'] = data_train['text'].apply(lambda x : Stemmer(x, lancaster))
data_test['text'] = data_test['text'].apply(lambda x : Stemmer(x, lancaster))        


# In[ ]:


def majNum(sentence):
    sentence=word_tokenize(sentence) 
    s=0
    for word in sentence:
        if(len(re.findall("[A-Z]{3,}", word))):
            s=s+1
    return  s
#Majuscule word
data_train['majusculenumber']=data_train['text'].apply(majNum)
data_test['majusculenumber']=data_test['text'].apply(majNum)


# In[ ]:


data_train['text'] = data_train['text'].apply(lambda x: x.lower())
data_test['text'] = data_test['text'].apply(lambda x: x.lower())


# In[ ]:


#Hashing : Tokenization - Train 
response = vectorizer.fit_transform(data_train['text'].values)
data_train = data_train.reset_index()
data_transform_train = pd.concat([data_train.drop(['sentiment', 'text'], axis = 1),pd.DataFrame(response.todense())], axis = 1)


# In[ ]:


#X And Y - Train
X_train = data_transform_train.copy()
Y_train = data_train[['sentiment']].copy()
Y_train.astype('int')
X_train.shape, Y_train.shape


# In[ ]:


#Hashing : Tokenization - Test
response = vectorizer.transform(data_test['text'].values)
data_test = data_test.reset_index()
# PER TESTARE IL CODICE SULL'EVALUATION DATAFRAME 
data_transform_test = pd.concat([data_test.drop(['ids','text'], axis = 1),pd.DataFrame(response.todense())], axis = 1)


# In[ ]:


#X and Y Test
X_test = data_transform_test.copy()
X_test.shape


# In[ ]:


#Logistic Regression MAX ENTROPY]
grid={"C":[5, 10, 15], "penalty":["l1","l2"]}
logreg = LOgisticRegression(solver = 'liblinear')
logreg_cv=GridSearchCV(logreg,grid,scoring = 'f1_macro', cv=10)


# In[ ]:


logreg_cv.fit(X_train.values,Y_train.values.reshape(-1))


# In[ ]:


res_logreg = logreg_cv.predict(X_test)
df_new = pd.concat([data_test.loc[:, ['ids']], pd.DataFrame(res_logreg)], axis=1)
df_new =  df_new.rename({'ids': 'Id', 0: 'Predicted'}, axis=1)
df_new.loc[:, ['Id']] = df_new.index 
results_path = 'results_logreg.csv' 
df_new.to_csv(results_path, index = False)


# In[ ]:


logreg_cv.best_params_


# In[ ]:


#Linear SVC
from sklearn.svm import LinearSVC
grid={"C":[5, 10, 15], "penalty":["l1","l2"]}
svc = LinearSVC()
svc_cv=GridSearchCV(svc, grid,scoring = 'f1_macro', cv=10)


# In[ ]:


svc_cv.fit(X_train.values,Y_train.values.reshape(-1))


# In[ ]:


logreg_cv.best_params_, svc_cv.best_params_


# In[ ]:


res_svc = svc_cv.predict(X_test)
df_new = pd.concat([data_test.loc[:, ['ids']], pd.DataFrame(res_svc)], axis=1)
df_new =  df_new.rename({'ids': 'Id', 0: 'Predicted'}, axis=1)
df_new.loc[:, ['Id']] = df_new.index 
results_path = 'results_svc.csv' 
df_new.to_csv(results_path, index = False)


# In[ ]:


logreg_cv.cv_results_['mean_test_score']


# In[ ]:


svc_cv.cv_results_['mean_test_score']


# In[ ]:


height_lr_l1 = [0.77939768, 0.77663156, 0.77529593]
height_lr_l2 = [0.36419001, 0.36419001, 0.36419001]
height_svc_l2 = [0.48446444, 0.48332909, 0.48763798]
# Histogram of tweets length
fig, ax = plt.subplots(figsize=(8,6))
x = [5, 10, 15]
ax.plot(x, height_lr_l1, color='tab:orange', marker='x', label='LR penalty l1')
ax.plot(x, height_lr_l2, color='tab:green', marker="^", label='LR penalty l2')
ax.plot(x, height_svc_l2, color='tab:blue', marker='o', label='SVC penalty l2')
ax.set_ylabel('macro f1 score', fontsize = 13)
ax.set_xlabel('C value', fontsize = 13)
ax.legend(fontsize = 13)
plt.show()
fig.savefig("f1scores.pdf")

