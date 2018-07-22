# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:45:44 2018

@author: hp-pc
"""

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import nltk
except ImportError as e:
    print(e)
    
# nltk.download('stopwords')
#from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#
## Tokenizing sentences
#dataset = nltk.sent_tokenize(paragraph)
#    
#    
##Bag of words
#from sklearn.feature_extraction.text import CountVectorizer
#cv= CountVectorizer(max_features=1500)#You can give stopword here itesef no need to do above steps if we provide the parameter
##Argument max_feature give only the most frequent 1500 words
#X=cv.fit_transform(corpus).toarray()
##To get the feature names
#cv.get_feature_names()
#
#first = cv.transform(["wow love place"]).toarray()
#
#
##Just to show
#ms0=corpus[0]
#ms0 = cv.transform([ms0])
#print(ms0)
#
#
##Create y
#
#y=dataset.iloc[:,1].values
#
#
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)
#
##Create model for Bayes regression
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train,y_train)
#y_pred=classifier.predict(X_test)
#
##Creating Confusion matrix
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)
#from sklearn.metrics import accuracy_score,classification_report
#acc=accuracy_score(y_test,y_pred)
#  
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train)
#accuracies.mean()
#accuracies.std()
import re
def preprocessed_data(tweet):
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    return tweet
try:
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    dataset = pd.read_csv('data.csv',usecols= [2,3])
#    X = dataset.iloc[:,0].copy().tolist()
#    y= dataset.iloc[:,1].copy()
#    dataset['Tweet']=list(map(preprocessed_data,X))
    from sklearn.model_selection import train_test_split
#    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)
    
    X_train, X_test, y_train, y_test = train_test_split(dataset['Tweet'], dataset['ADR_label'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#    from sklearn.feature_extraction.text import CountVectorizer
#    count_vect = CountVectorizer()
#    X_train_counts = count_vect.fit_transform(X_train)
    
#    from sklearn.feature_extraction.text import TfidfVectorizer
#    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#    X_train_tfidf = tfidf.fit_transform(X_train)
    
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    predicted = clf.predict(count_vect.transform(X_test))
    
    np.mean(predicted == y_test)
    
except Exception as e:
    priint(e)