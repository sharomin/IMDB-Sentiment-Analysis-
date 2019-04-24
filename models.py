# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.svm import LinearSVC
##### Train Foder #############################################################
import preprocessing as prepro
X_train, X_test, y_train, y_test = prepro.inputresults()

##### Test Folder #############################################################
#import test_loader as test_loader
#X_train, X_test, y_train, y_test, X_test_folder = test_loader.inputresults()#go to preprocessing.py and add the source of the file 
#y_test_folder = naiveBayesSVM(X_train, X_test_folder, y_train)
##convert_to_csv(y_test_folder)
#y_test_folder_dataframe = pd.DataFrame(y_test_folder)
#y_test_folder_dataframe.to_csv('results_1.csv',index=True, header=True)

############################# MODELS ##########################################

class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):

    def __init__(self, alpha=2, C=3, beta=1, fit_intercept=False):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            coef_, intercept_ = self._fit_binary(X, y)
            self.coef_ = coef_
            self.intercept_ = intercept_
        else:
            coef_, intercept_ = zip(*[
                self._fit_binary(X, y == class_)
                for class_ in self.classes_
            ])
            self.coef_ = np.concatenate(coef_)
            self.intercept_ = np.array(intercept_).flatten()
        return self

    def _fit_binary(self, X, y):
        p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
        q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
        r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())
        b = np.log((y == 1).sum()) - np.log((y == 0).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix(
                (r, (indices, indices)),
                shape=(len(r), len(r))
            )
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

        lsvc = LinearSVC(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=10000
        ).fit(X_scaled, y)

        mean_mag =  np.abs(lsvc.coef_).mean()

        coef_ = (1 - self.beta) * mean_mag * r + \
                self.beta * (r * lsvc.coef_)

        intercept_ = (1 - self.beta) * mean_mag * b + \
                     self.beta * lsvc.intercept_

        return coef_, intercept_
      
# compare all models
def conv_matrix(y_test, y_pred):
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    Conf_mat = confusion_matrix(y_test, y_pred)
    ACC_score = accuracy_score(y_test, y_pred)
    return Conf_mat, ACC_score

#NaiveBayesSVM 
def naiveBayesSVM(X_train, X_test, y_train):
    # Fitting Kernel SVM to the Training set
    from sklearn.metrics import roc_auc_score
    classifier = NBSVM()
    classifier.fit(X_train, pd.Series(y_train))
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred, classifier 
#SVM
def SVM(X_train, X_test, y_train):
    # Fitting Kernel SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred, classifier
#Logestic Regression 
def logestic_reg(X_train, X_test, y_train):
    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred, classifier
#RandomForest
def random_forest(X_train, X_test, y_train):
    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred, classifier
#Bernoli Naive Bayes
def bernoli_naive_bayes(X_train, X_test, y_train):
    # Fitting bernoli Naive Bayes to the Training set
    from sklearn.naive_bayes import BernoulliNB
    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred, classifier

#stochastic GD
def stoch_GD():
    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier(alpha =1e-5 , epsilon =0.1 )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred
        
def gauss_naive(X_train, X_test, y_train):
    # Fitting gaussian Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred



#RigidClassifier
def rigid(X_train, X_test, y_train):
    # Fitting RigidClassifier to the Training set
    from sklearn.linear_model import RidgeClassifier
    classifier = RidgeClassifier(alpha=4, class_weight='balanced')
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

#def svm(X_train, X_test, y_train):
#    # Fitting Kernel SVM to the Training set
#    from sklearn.svm import LinearSVC
#    classifier = LinearSVC(random_state = 0)
#    classifier.fit(X_train, y_train)
#    # Predicting the Test set results
#    y_pred = classifier.predict(X_test)
#    return y_pred

def knn(X_train, X_test, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train,y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def descision_tree(X_train, X_test, y_train):
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred



def xg_booster(X_train, X_test, y_train):
    # Fitting XGBoost to the Training set
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def k_fold_cross(classifier, X_train, y_train):
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    mean = accuracies.mean()
    std = accuracies.std()
    return accuracies, mean, std 

def grid_search(classifier, X_train, y_train):
    # Applying Grid Search to find the best model and the best parameters
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    return best_accuracy, best_parameters


# =============================================================================
NB_SVM_y_pred, NB_SVM_classifier = naiveBayesSVM(X_train, X_test, y_train)
NB_SVM_Conf_mat, NB_SVM_ACC_score = conv_matrix(y_test, NB_SVM_y_pred)
NB_SVM_accuracies, NB_SVM_mean, NB_SVM_std = k_fold_cross(NB_SVM_classifier, X_train, pd.Series(y_train))
# =============================================================================

# =============================================================================
LR_y_pred, LR_classifier = logestic_reg(X_train, X_test, y_train)
LR_Conf_mat, LR_ACC_score = conv_matrix(y_test, LR_y_pred)
LR_accuracies, LR_mean, LR_std = k_fold_cross(LR_classifier, X_train, pd.Series(y_train))
# =============================================================================

# =============================================================================
RF_y_pred, RF_classifier = random_forest(X_train, X_test, y_train)
RF_Conf_mat, RF_ACC_score = conv_matrix(y_test, RF_y_pred)
RF_accuracies, RF_mean, RF_std = k_fold_cross(RF_classifier, X_train, pd.Series(y_train))
# =============================================================================

# =============================================================================
BNB_y_pred, BNB_classifier = bernoli_naive_bayes(X_train, X_test, y_train)
BNB_Conf_mat, BNB_ACC_score = conv_matrix(y_test, BNB_y_pred)
BNB_accuracies, BNB_mean, BNB_std = k_fold_cross(BNB_classifier, X_train, pd.Series(y_train))
# =============================================================================

# =============================================================================
SVM_y_pred, SVM_classifier = SVM(X_train, X_test, y_train)
SVM_Conf_mat, SVM_ACC_score = conv_matrix(y_test, SVM_y_pred)
SVM_accuracies, SVM_mean, SVM_std = k_fold_cross(SVM_classifier, X_train, pd.Series(y_train))
SVM_best_accuracy, SVM_best_parameters = grid_search(SVM_classifier, X_train, y_train)
# =============================================================================

# =============================================================================    
 #pipeline
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
stop_words = set(stopwords.words('english'))
import preprocessing as prepro
X_train, X_test, y_train, y_test = prepro.inputresults_grid()
 # build the pipeline
ppl = Pipeline([
               ('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf',   LogisticRegression())
       ])
 
    
params = {"vect__ngram_range": [(1,2)],
           "vect__binary" : [True, False],                     
           "vect__max_df": [0.8],
           "vect__stop_words" : [stop_words],
           "vect__min_df" : [5],
           "vect__max_features": [3000],
           "tfidf__use_idf": [True],
           "clf__C" : [3, 4]}
  
grid = GridSearchCV(ppl, param_grid= params, n_jobs= -1, cv=10)
grid.fit(X_train, y_train)
Best_cross_validation_score = grid.best_score_
Best_parameters = grid.best_params_
Best_estimator = grid.best_estimator_
acc_Best_estimator = accuracy_score(y_test, Best_estimator.predict(X_test)) 

 
# =============================================================================
