# IMDB-Sentiment-Analysis-
In this project, I have developed models to predict the sentiment of IMBD reviews. IMDB is a popular website and database of movie information and reviews (https://www.imdb.com/). The goal is to classify IMBD reviews as positive or negative based on the language they contain. This project was done for competing with other groups in www.kaggle.com to achieve the best accuracy in a competition.
# How to run :
Tor run All models results:
preprocessing.py : data cleaning is done def text_preprocessing() and results are saved as pickles "processed_pos"/"processed_neg" using the function in pickler.py
        

1)In models.py  
- run models.py to get X_train, X_test, y_train, y_test , 
        X_train, X_test, y_train, y_test = prepro.inputresults()
- run each model objesls :
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

you can see all models results and Grird search for SVM 
- pipeline for logestic Reggression  :
- run : 
        Best_cross_validation_score = grid.best_score_
        Best_parameters = grid.best_params_

To run Bernoulli Naive Bayes:
1) In pickler.py: 
- run:
    - split_training_set_as_test_X()

2) In bernoulli_naive_bayes.py:
- run:
    - test_X = pkler.pkl_load( “split_train" )
    - test_X, true_y, pos_counts, neg_counts, train_X = pkler.prep_predict_y(test_X)
    - y_preds = predict_test_y( test_X, pos_counts, neg_counts, train_X )
    - prep.convert_to_csv( y_preds, "y_preds" )
    - prep.convert_to_csv( true_y, "true_y" )
