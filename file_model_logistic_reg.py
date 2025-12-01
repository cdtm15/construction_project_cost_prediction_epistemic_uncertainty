#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:57:56 2024

@author: cristiantobar
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt


def model_logistic_reg(X_train, X_test, y_train, y_test, X_encoded):
    n_train = len(X_train)
    n_test  = len(X_test)
        
    feature_train = X_train.iloc[:,0:3]
    feature_test  = X_test.iloc[:,0:3]
    
    log_reg = LogisticRegression()
    log_reg.fit(feature_train, y_train)
    
    n_feat = feature_train.shape[1]
    
    y_predict = log_reg.predict(feature_test)
        
    log_reg.score(feature_test, y_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    accu           = (tp+tn)/(tp+fp+fn+tn)
    sensi          = (tp)/(tp+fn)
    speci          = (tn)/(tn+fp)
    f1             = (2*tp)/((2*tp)+fp+fn)
    mse            = mean_squared_error(y_test, y_predict)
    r2             = r2_score(y_test, y_predict)
    
    
    actual_num_feat = feature_train.shape[1];
    
    # List of coeficient values
    coefs = np.union1d(log_reg.intercept_, log_reg.coef_[0]).tolist()
    # List of names
    betas = ['beta_' + str(i) for i in range(len(coefs))]
    # Combine `coefs` & `betas` to form a dictionary
    d = dict(zip(betas, coefs))
    # Print as formula
    print('L(' + str(d) + ')')

    return [n_feat, n_train, n_test, r2, accu, sensi, speci, f1, mse]
    
    
    
    
    
    