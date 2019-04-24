import pickle
import os
import csv
from pprint import pprint
import numpy as np
from sklearn.model_selection import train_test_split

def pos_pkl_save():         # FORMAT: { f# : string }
    pos_samples = {}
    dir = "../comp-551-imbd-sentiment-classification/train/pos/" # SET TO YOURS

    for sample in os.listdir(dir):
        file = open(dir + sample, "r")
        str = ""
        for line in file:
            str += line

        pos_samples[sample[:-4]] = str

    # Save to pickle
    with open('pos.pickle', 'wb') as handle:
        pickle.dump(pos_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pkl_save(to_save, pickle_name):
    pkl_open = pickle_name + ".pickle"
    with open(pkl_open, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved to pickle as " + pickle_name + ".pickle")

def neg_pkl_save():
    neg_samples = {}
    dir = "../comp-551-imbd-sentiment-classification/train/neg/"  # SET TO YOURS

    for sample in os.listdir(dir):
        file = open(dir + sample, "r")
        str = ""
        for line in file:
            str += line

        neg_samples[sample[:-4]] = str

    # Save to pickle
    with open('neg.pickle', 'wb') as handle:
        pickle.dump(neg_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test_pkl_save():
    # test_X stored as: ordered 2D array: [ [k1, v1], [k2, v2], [k3, v3], ... ]
    # Where k are the file names and v are the strings

    test_X = {}                                                 # First, save to dictionary
    dir = "../comp-551-imbd-sentiment-classification/test/"     # Extracting X (strings)
    for sample in os.listdir(dir):
        file = open(dir + sample, "r")
        str = ""
        for line in file:
            str += line
        test_X[sample[:-4]] = str                               # [:-4] to remove ".txt"

    to_int = { int(k) : v for k, v in test_X.items() }          # Convert all keys to integers
    test_X_array = []                                           # New empty list
    for k in sorted(to_int):
        temp = []
        temp.append(k)
        temp.append(to_int[k])
        test_X_array.append(temp)

    # Save to pickle
    with open('test_array.pickle', 'wb') as handle:
        pickle.dump(test_X_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pkl_load(filename):
    # Load from pickle
    filename += ".pickle"
    with open(filename, 'rb') as handle:
        unpickled = pickle.load(handle)
    return unpickled

def extract_test_X():
    # test_X stored as: (dictionary)
    #   { key: file# (0,1,2), value: str (from test_X) }
    test_X = {}
    dir = "../comp-551-imbd-sentiment-classification/test/"     # Extracting X (strings)
    for sample in os.listdir(dir):
        file = open(dir + sample, "r")
        str = ""
        for line in file:
            str += line
        test_X[sample[:-4]] = str                               # [:-4] to remove ".txt"

    # Save to pickle
    with open('test.pickle', 'wb') as handle:
        pickle.dump(test_X, handle, protocol=pickle.HIGHEST_PROTOCOL)

def split_training_set_as_test_X():
    # Returns as 2D array: [ f#, string, actual y ]
    train_test_X = []

    dir = "../comp-551-imbd-sentiment-classification/train/pos/"
    for sample in os.listdir(dir):
        temp = []
        file = open(dir + sample, "r")
        str = ""
        for line in file:   str += line
        temp.append(sample[:-4])
        temp.append(str)
        temp.append(1)                                  # 1 for pos
        train_test_X.append(temp)

    dir = "../comp-551-imbd-sentiment-classification/train/neg/"
    for sample in os.listdir(dir):
        temp = []
        file = open(dir + sample, "r")
        str = ""
        for line in file:   str += line
        temp.append(sample[:-4])
        temp.append(str)
        temp.append(0)                                  # 0 for neg
        train_test_X.append(temp)

    with open('split_train.pickle', 'wb') as handle:
        pickle.dump(train_test_X, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Counting instances of x_i when y = 1 or y = 0
def calc_counts_helper(pn):
    counts = {}
    for p in pn:
        temp = dict((x,p.count(x)) for x in set(p.split()))
        for key, value in temp.items():
            if key not in counts:
                counts[key] = value
            if key in counts:
                counts[key] += value
    return counts

def calc_counts(pos, neg):
    pos_counts = calc_counts_helper(pos)
    neg_counts = calc_counts_helper(neg)

    return pos_counts, neg_counts

def test_text_preprocessing(test_x):           # test_x = list of strings
    import re
    import nltk
    # nltk.download('stopwords')                      # Comment/uncomment
    from nltk.corpus import stopwords               # Removing stopwords
    from nltk.stem.porter import PorterStemmer      # Lemmatization
    #nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer

    corpus = []
    for sample in test_x:
        raw_text = str(sample)
        br = "<br />"
        lem_text = re.sub(br,' ', raw_text)                  # Remove breaks from text
        lem_text = re.sub('[^a-zA-Z]', ' ', lem_text)     # Remove any character that isn't alphabet
        lem_text = re.sub(r'\W', ' ', lem_text)
        lem_text = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', ' ', lem_text)
        lem_text = re.sub(r'<[^<>]+>', " ", lem_text)                     # Removing all the tags
        lem_text = re.sub(r'[0-9]+', ' ', lem_text)                       # Removing all the numbers
        lem_text = re.sub(r'[?|!|\'|"|#]',r' ',lem_text)                  # Removing all puncs
        lem_text = re.sub(r'[.|,|)|(|\|/]',r' ',lem_text)
        lem_text = re.sub(r'[^\w\s]','',lem_text)
        lem_text = re.sub(r'\^[a-zA-Z]\s+', ' ', lem_text)                # Removing single characters from the beginning
        lem_text = re.sub(r'\s+', ' ', lem_text, flags=re.I)              # Substituting multiple spaces with single space
        lem_text = re.sub(r'^b\s+', ' ', lem_text)                        # Removing prefixed 'b'
        lem_text = lem_text.lower()                       # Lowercase all
        lem_text = lem_text.split()                       # Split
        lemmatizer = WordNetLemmatizer()                  # Lemmatization of the data
        lem_text = [lemmatizer.lemmatize(word) for word in lem_text]
        lem_text = ' '.join(lem_text)
        lem_text = lem_text.split()
        lem_text = [word for word in lem_text if len(word) > 2]
        lem_text = ' '.join(lem_text)
        corpus.append(lem_text)
    return corpus

def cheap_solution(sentence):
    sentence_arr = []
    sentence_arr.append(str(sentence))
    lem_sent_arr = test_text_preprocessing(sentence_arr)
    lem_sentence = lem_sent_arr[0]
    return lem_sentence

def split_80_20(arr):
    arr = np.array(arr)                                                         # Convert to numpy array
    X, y = np.split(arr, [-1], axis = 1)                                        # Separate X and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test                                     # Train = 80%, Test = 20%

def prep_predict_y(test_X):
    print("Preparing for prediction:")
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = split_80_20(test_X)

    # test_X:
    # FORMAT: { key: file# (0,1,2), value : (raw) str }
    test_X = {}
    for Xy in X_test:
        test_X[Xy[0]] = Xy[1]

    # true_y:
    # FORMAT: { key: file# (0,1,2), value : true y }
    true_y = {}
    for i in range(len(X_test)):
        true_y[X_test[i][0]] = np.squeeze(y_test)[i]

    # pos_counts, neg_counts:
    # FORMAT: { word : # instances in pos/neg }
    pos = []
    neg = []
    y_train_squeeze = np.squeeze(y_train).tolist()
    X_train = X_train.tolist()
    for i in range(len(y_train_squeeze)):
        Xy = X_train[i]
        if y_train_squeeze[i] == '0':     neg.append(Xy)
        elif y_train_squeeze[i] == '1':   pos.append(Xy)

    pos = test_text_preprocessing(pos)                                     # Lem the strings
    neg = test_text_preprocessing(neg)

    pos_counts, neg_counts = calc_counts(pos, neg)                         # Count instances of word

    # train_X:
    # FORMAT: { (lem) string : y }
    train_X = {}
    train_merged = np.concatenate((X_train, y_train), axis = 1)
    etc, str_y = np.split(train_merged, [-2], axis = 1)
    str_y = np.squeeze(str_y)

    for i in range(len(str_y)):
        sentence = str_y[i][0]
        lem_sentence = cheap_solution(sentence)
        y_val = str_y[i][1]
        train_X[lem_sentence] = y_val

    return test_X, true_y, pos_counts, neg_counts, train_X


# # Run this to create the pickle on your computer
# split_training_set_as_test_X()

# # Run this to load the pickle
# test_X = pkl_load("split_train")                # test_X is a 2D array
# test_X, true_y, pos_counts, neg_counts, train_X = prep_predict_y(test_X)
