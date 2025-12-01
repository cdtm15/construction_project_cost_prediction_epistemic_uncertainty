#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:40:18 2024

@author: cristiantobar
"""

# Import necessary modules 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import matplotlib.pyplot as plt 

def model_knn(X_train, X_test, y_train, y_test, X_encoded):
    
    n_train = len(X_train)
    n_test  = len(X_test)
    
    neighbors = np.arange(1, 20) 
    train_accuracy = np.empty(len(neighbors)) 
    test_accuracy = np.empty(len(neighbors)) 
      
    # Loop over K values 
    for i, k in enumerate(neighbors): 
        knn = KNeighborsClassifier(n_neighbors=k) 
        knn.fit(X_train, y_train) 
          
        # Compute training and test data accuracy 
        train_accuracy[i] = knn.score(X_train, y_train) 
        test_accuracy[i] = knn.score(X_test, y_test) 
      
    # # Generate plot 
    # plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 
    # plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 
    # plt.legend() 
    # plt.xlabel('n_neighbors') 
    # plt.ylabel('Accuracy') 
    # plt.show() 
    
    max_idx = np.argmax(test_accuracy)
    max_val = test_accuracy[max_idx]
    
    best_k = max_idx + 1
    
    knn = KNeighborsClassifier(n_neighbors=best_k) 
    knn.fit(X_train, y_train) 
    
    n_feat  = X_train.shape[1]

    
    y_predict = knn.predict(X_test)
        
    knn.score(X_test, y_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    accu           = (tp+tn)/(tp+fp+fn+tn)
    sensi          = (tp)/(tp+fn)
    speci          = (tn)/(tn+fp)
    f1             = (2*tp)/((2*tp)+fp+fn)
    mse            = mean_squared_error(y_test, y_predict)
    r2             = r2_score(y_test, y_predict)
    
    return [n_feat, n_train, n_test, r2, accu, sensi, speci, f1, mse]