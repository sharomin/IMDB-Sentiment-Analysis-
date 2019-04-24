import numpy as np
import math
import os
import preprocessing as prep
from pprint import pprint
import pickler as pkler
import pickle
import csv

# NAIVE BAYES:  P(y = 1 | x) = P(x|y = 1) * P(y = 1) / P(x) <-- ignoring P(x)
#
# P(x_i|y = 1)  MLE: prob of seeing word i in samples of class 1 (1 = positive samples)
#               how likely are we to see the observed features if the point was from class 1?
#               = [(number of instances x_i = 1 when y = 1) + 1] / [(# of examples y = 1) + 2]
#               including LAPLACE SMOOTHING
#               = [(number of instance x_i = 1 and y = 1) + 1] / [(number of examples with y = 1) + 2]
# P(y = 1)      MARGINAL PROBABILITY: ignoring features, how likely are we to see class 1 (positive)?

# Helper: used for both marginal and MLE calculations
def y_counts(y):
    train_y = []
    for k,v in y.items():
        train_y.append(v)

    freqs = dict((x,train_y.count(x)) for x in set(train_y))
    num_0, num_1 = freqs['0'], freqs['1']
    return num_0, num_1

# Returns P(y = 1) and P(y = 0)
def calc_marginal(train_X):
    num_0, num_1 = y_counts(train_X)
    total = num_0 + num_1
    num_0 = num_0/total
    num_1 = num_1/total
    return num_0, num_1

# Returns maximum likelihood estimator (MLE) as dict after laplace smoothing
# MLE per word in the sentence (on a small scale, sum is done in calculate_log_odds method)
# MLE = (num instances x_j = 1 when y = 0,1 (whatever input) + 1)
#       / ((num examples y = input) + 2)
# inputs:       word - the word to be searched in pos/neg_counts dictionary
#               klass - either 1 (pos), 0 (neg)
#               counts - corresponds to klass (pos_counts or neg_counts)
def calc_mle(word, klass, counts, train_X):
    if word in counts: num_instances_x_i = float(counts[word]) + 1              # +1 for laplace smoothing
    else:              num_instances_x_i = 1
    num_0, num_1 = y_counts(train_X)                                            # using helper to get counts
    num_ex_y = 0
    if klass == 0:  num_ex_y = float(num_0 + 2)                                 # set which num_0/1 is num_ex_y
    if klass == 1:  num_ex_y = float(num_1 + 2)                                 # +2 for laplace smoothing

    mle = float(num_instances_x_i / num_ex_y)

    return mle
    # this could be done without calling helper twice, but I wanted to make it more clear what was happening

# Log odds = log( P(y=1|x) / P(y=1|x) )
#          = log( theta_1 / (1 - theta_1) ) + sum[from j = 1 to m](log( P(x_j|y=1) / P(x_j|y=0) ) )
def calculate_log_odds(sentence, train_X, pos_counts, neg_counts):
    theta_0, theta_1 = calc_marginal(train_X)
    log_odds = math.log(theta_1/(1 - theta_1))

    # For every feature, calculate mle
    sum = 0
    print(sentence)
    for word in sentence.split():                     # For every word in the sentence
        calc = math.log( calc_mle(word, 1, pos_counts, train_X) / calc_mle(word, 0, neg_counts, train_X) )  # Could be done without calling helper twice, but wanted to make it more clear what was happening
        sum += calc
    log_odds += sum

    return log_odds

# Prediction: using mle, marginals
def predict_test_y(test_X, pos_counts, neg_counts, train_X):
    # First, check if test X have already been preprocessed (lemmatized).

    print(len(test_X))

    y_preds = {}                                                                # Format: { file_num : predicted y (0/1) }
    print("Calculating log odds:")
    i = 0
    for file_n in test_X:                                                       # Where { file_n : lemmed string }
        print(i)
        i+=1
        log_odds = calculate_log_odds(                                          # Want to find the log odds for every SENTENCE = test_X[file_n]
            test_X[file_n],
            train_X,
            pos_counts,
            neg_counts)
        print(log_odds)
        if log_odds > 0:    y_preds[file_n] = 1                                 # Predict as positive
        else:               y_preds[file_n] = 0                                 # Predict as negative

    with open('training_y_preds.pickle', 'wb') as handle:
        pickle.dump(y_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return y_preds

# =================================== main ==========================================
# # Setting up (PREPROCESSED) X and y from preprocessing.py:
# pos, neg = prep.unpickle_individually()                         # FORMAT: pos, neg are LISTS of (lem) STRINGS
#
# # Processing step
# pos_counts, neg_counts = calc_counts(pos, neg)                  # FORMAT: { word : # instances in pos/neg }
# train_X = prep.create_y_and_combine(pos, neg)                   # FORMAT: { (lem) string : y }
#
# # Prediction step
# # test_X = pkler.pkl_load("test")                               # Load test X from pickle. FORMAT: { key: file# (0,1,2), value : (raw) str }
# # y_preds = predict_test_y(                                     # Predict test y's
# #                 test_X,
# #                 pos_counts,
# #                 neg_counts,
# #                 train_X)

test_X = pkler.pkl_load("split_train")
test_X, true_y, pos_counts, neg_counts, train_X = pkler.prep_predict_y(test_X)

y_preds = predict_test_y(                                         # FORMAT: { file_num : predicted y (0/1) }
                test_X,
                pos_counts,
                neg_counts,
                train_X)

# Convert dictionary to csv file (results.csv)
prep.convert_to_csv(y_preds, "y_preds")
prep.convert_to_csv(true_y, "true_y")

# Compare manually first
