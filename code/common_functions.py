# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 15:36:03 2014

@author: ogavril

PURPOSE: functions used by multiple programs and other ***_function.py

OUTPUT:  none 
"""

import numpy as np
import math
import pandas as pd

##### data prep functions ########################
def make_dataMatrix_fromDF(columns,train_df,normalizeInput=True):
    nps = np.array([])
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = normalize(train_df[columns[c]])
        else:
            norm_vec = train_df[columns[c]]
        nps = np.append(nps,norm_vec) #train_df[columns[c]])
    nps = nps.reshape((len(columns),len(train_df.index)))
    train_data = nps.transpose()
    return train_data
    
    
def make_data_4scikit_functions(columns,train_df,test_df,target_name,normalizeInput=True):
    nps = np.array([])
 
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = normalize(train_df[columns[c]])
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
            norm_vec = normalize(test_df[columns[c]])
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
    

def def_cross_validation_subsets(df,varN,numK=5):
    df[varN] = -1
    for i in xrange(len(df.index)):
        df[varN].iloc[i] = i%numK
    return df

def silly_cuberoot(col):
    col1 = []
    for elem in col:
        if elem < 0:
            col1.append(-1*math.pow(-elem,1./3.))
        else:
            col1.append(math.pow(elem,1./3.))
    col1 = np.array(col1)
    return col1

def MSE(predicted_vals,true_vals):
    a = predicted_vals - true_vals
    mse = np.dot(a,a)/float(len(a))
    return mse
    
    
def AbsError(predicted_vals,true_vals):
    a = np.abs(predicted_vals - true_vals)
    return np.mean(a)

def determine_correlation(var1,var2):
    """assumes NaNs have been dropped"""
    v1 = np.array(var1)
    v2 = np.array(var2)
    mat = np.c_[(v1,v2)]# np.vstack((v1,v2)) #
    corr = np.corrcoef(mat.T)
    return corr[0][1]
    
def normalize(vec):
    """i think there is an sp.linalg.norm function, but for some reason it's not working for me possibly because 
    I don't require that vec is an np.array """
    min_ = np.min(vec)
    max_ = np.max(vec)
    if min_ != max_:
        n_vec = (vec-min_)/(max_-min_)
        return n_vec

    return vec
