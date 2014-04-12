# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:47:25 2014

@author: ogavril

scikit learn functions for machine learning etc.
"""

import numpy as np
from sklearn import svm,tree

from sklearn import ensemble# import RandomForestClassifier, ExtraTreesClassifier

import common_functions
from class_definitions import *


def make_data_4scikit_functions(columns,train_df,test_df,target_name,normalizeInput=True):
    nps = np.array([])
 
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = common_functions.normalize(train_df[columns[c]])
        else:
            norm_vec = train_df[columns[c]]
        nps = np.append(nps,norm_vec) #train_df[columns[c]])
    nps = nps.reshape((len(columns),len(train_df.index)))
    train_data = nps.transpose()
    train_target = np.array(train_df[target_name])
#    print train_data.shape
#    print train_target.shape
    
    nps = np.array([])
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = common_functions.normalize(test_df[columns[c]])
        else:
            norm_vec = test_df[columns[c]]
        nps = np.append(nps,norm_vec) #test_df[columns[c]])
    nps = nps.reshape((len(columns),len(test_df.index)))
    test_data = nps.transpose()   
    try: 
        test_target = np.array(test_df[target_name])     
    except:
        print "!!! CAUTION:",target_name,"does not exist for test data...okay for final prediction"
        test_target = None
        
    return train_data,train_target,test_data,test_target     

def perform_svm(train_data,train_target,test_data,test_target,kernel,polydeg=3):
    if kernel == 'poly':
        mod = svm.SVC(C=3.0,kernel=kernel,degree=polydeg)
    else:
        mod = svm.SVC(C=10.0,kernel=kernel,gamma=0.9)
    mod.fit(train_data,train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test

def perform_CART(train_data,train_target,test_data,test_target, DT_max_depth=None, DT_max_features=None,DT_random_state=None):
    mod = tree.DecisionTreeClassifier(max_depth=DT_max_depth, max_features=DT_max_features,random_state=DT_random_state)
    #print mod
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test

def perform_RandomForest(train_data,train_target,test_data,test_target,num_trees=10, RF_max_depth=None, RF_max_features=None,RF_random_state=None,classifier='RandomForest'):
    if classifier.lower()== 'randomforest':    
        mod = ensemble.RandomForestClassifier(n_estimators = num_trees,max_depth=RF_max_depth, max_features=RF_max_features,random_state=RF_random_state)
    else:
        mod = ensemble.ExtraTreesClassifier(n_estimators = num_trees,max_depth=RF_max_depth, max_features=RF_max_features,random_state=RF_random_state)
    #print mod n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test

def summarize_model_stats(model_stats,stats):
    train_sets = model_stats.keys()
    kernels = []
    for trs in model_stats.keys():
        for kernel in  model_stats[trs].keys():
            if kernel not in kernels:
                kernels.append(kernel)
    print train_sets
    print kernels
    models = []

    for kernel in kernels:
        model = DISCRETE_MODEL(kernel)
        for stat in stats:
            t,count = 0,0
            for trs in train_sets:
                if kernel in model_stats[trs].keys():
                    t += model_stats[trs][kernel][stat]
                    count += 1
            model.__dict__[stat] = t/float(count)
        models.append(model)
    return models
        
def calculate_accuracy(target_predicted, target):
    """function to use for SVM, CART, etc. where np arrays are passed to determine who different they are"""
    if len(target_predicted) == len(target):
        diff = np.abs(target_predicted - target)
        num_correctly_predicted = len(diff) - np.count_nonzero(diff)
        accuracy = float(num_correctly_predicted)/float(len(diff))
        return accuracy
    else:
        print "\n!!! ERROR in",calculate_accuracy.__name__
        return None    



