# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import re
import nltk
import pickle
import pickler as pkler
# nltk.download('stopwords')                      # Comment/uncomment
from nltk.corpus import stopwords               # Removing stopwords
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


def unpickle(): # Returns positive and negative samples as two separate arrays
    pos_processed_corpus = pkler.pkl_load("processed_pos")      # POS: Load from pickle
    neg_processed_corpus = pkler.pkl_load("processed_neg")      # NEG: Load from pickle
    all_samples = pos_processed_corpus + neg_processed_corpus
    return all_samples


def create_y(): # Returns y as list of 1/0 depending on size of pos/neg
    temp_pos = [1 for i in range(12500)]
    temp_neg = [0 for i in range(12500)]
    y = temp_pos + temp_neg
    return y


def read_test(path):
    data = []
    folder = 'test/'
    for i in range(25000):
        with open (os.path.join(folder, str(i)+'.txt'), 'rb' ) as f :
            review = f.read().decode('utf-8').replace('\n', '').strip().lower()
            data.append([review, i])
    return data


def vectorizers(X_train_folder, X_test_folder): # Creating the Bag of Words and TF*IDF model, return as numpy array
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=7000,
                                 encoding='latin-1',
                                 lowercase=True,
                                 preprocessor=None,
                                 tokenizer=None,
                                 min_df=5,
                                 max_df=0.7,     
                                 ngram_range=(1, 2),    
                                 stop_words=stopwords.words('english'))  
    
    X_train_vect = vectorizer.fit_transform(X_train_folder).toarray() 
    X_test_vect = vectorizer.transform(X_test_folder).toarray() 
    
    
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfconverter = TfidfTransformer(norm='l2')  
    X_train_tfidf = tfidfconverter.fit_transform(X_train_vect).toarray() 
    X_test_tfidf = tfidfconverter.transform(X_test_vect).toarray() 
    
    return X_train_tfidf, X_test_tfidf                                        # Numpy array

def splitter(X,y):
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.17, random_state = 101)
    return X_train, X_test, y_train, y_test

def inputresults_test():
    test = read_test('/Users/saeedshoarayenejati/Documents/GitHub/ML-Project-2-Group-89/') 
    test = pd.DataFrame(test)
    #test = test.sort_values('1', ascending=True)
    X_test_folder = test[0]
    return X_test_folder

def inputresults():
    X_train_folder, X_test_folder = vectorizers(unpickle(),inputresults_test())
    X_train, X_test, y_train, y_test = splitter(X_train_folder, create_y())
    return X_train, X_test, y_train, y_test, X_test_folder





#X_train, X_test, y_train, y_test, X_test_folder = inputresults()







