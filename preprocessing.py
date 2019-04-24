# Natural Language Processing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


def text_preprocessing():
    # Importing the dataset from pickle
    samples = pkler.pkl_load("pos")                         # Positive
#    samples = pkler.pkl_load("neg")                         # Negative
#    samples = pkler.pkl_load("test")  
    corpus = []
    for sample in samples:
        raw_text = samples[sample]
        br = "<br/>"
        imdb_review = re.sub(br,' ', raw_text)                  # Remove breaks from text
        imdb_review = re.sub('[^a-zA-Z]', ' ', imdb_review)     # Remove any character that isn't alphabet
        imdb_review = re.sub(r'\W', ' ', imdb_review)
        imdb_review = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', ' ', imdb_review)    
        # Removing all the tags
        imdb_review = re.sub(r'<[^<>]+>', " ", imdb_review) 
        # Removing all the numbers
        imdb_review = re.sub(r'[0-9]+', ' ', imdb_review) 
        #Removing all puncs
        imdb_review = re.sub(r'[?|!|\'|"|#]',r' ',imdb_review)
        imdb_review = re.sub(r'[.|,|)|(|\|/]',r' ',imdb_review)
        imdb_review = re.sub(r'[^\w\s]','',imdb_review)

        #Removing single characters from the beginning
        imdb_review = re.sub(r'\^[a-zA-Z]\s+', ' ', imdb_review) 
        # Substituting multiple spaces with single space
        imdb_review = re.sub(r'\s+', ' ', imdb_review, flags=re.I)
        # Removing prefixed 'b'
        imdb_review = re.sub(r'^b\s+', ' ', imdb_review)
#        imdb_review = imdb_review.lower()                       # Lowercase all
        imdb_review = imdb_review.split()                       # Split
         # Lemmatization of the data
        lemmatizer = WordNetLemmatizer()
        imdb_review = [lemmatizer.lemmatize(word) for word in imdb_review]
        imdb_review = ' '.join(imdb_review)
        imdb_review = imdb_review.split()
        imdb_review = [word for word in imdb_review if len(word) > 2]
        imdb_review = ' '.join(imdb_review)
        corpus.append(imdb_review)
    pkler.pkl_save(corpus, "processed_pos")                     # POS: Save to pickle using method from pickler.py
#    pkler.pkl_save(corpus, "processed_neg")                     # NEG: Save to pickle using method from pickler.py
#    pkler.pkl_save(corpus, "processed_test")  


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


################ TRAIN
def vectorizers(X): # Creating the Bag of Words and TF*IDF model, return as numpy array
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1000,
                                 encoding='latin-1',
                                 lowercase=True,
                                 preprocessor=None,
                                 tokenizer=None,
                                 min_df=5,
                                 max_df=0.7,     
                                 ngram_range=(1, 2),    
                                 stop_words=stopwords.words('english'))  
    
    X_train_vect = vectorizer.fit_transform(X).toarray() 
#    X_test_vect = vectorizer.transform(X).toarray() 
    
    
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfconverter = TfidfTransformer(norm='l2')  
    X_train_tfidf = tfidfconverter.fit_transform(X_train_vect).toarray() 
#    X_test_tfidf = tfidfconverter.transform(X_test_vect).toarray() 
    
    return X_train_tfidf                                        # Numpy array

def splitter(X,y):
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.17, random_state = 101)
    return X_train, X_test, y_train, y_test


def inputresults():
    X_train, X_test, y_train, y_test = splitter(vectorizers(unpickle()), create_y())
    return X_train, X_test, y_train, y_test

def inputresults_grid():
    X_train, X_test, y_train, y_test = splitter(unpickle(), create_y())
    return X_train, X_test, y_train, y_test






