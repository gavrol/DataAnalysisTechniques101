# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 15:36:03 2014

@author: ogavril
"""

import numpy as np
import math

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
