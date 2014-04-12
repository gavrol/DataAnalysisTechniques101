# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 07:40:12 2014

@author: ogavril

PURPOSE: instead of using scikit for some of the pure stats things, let's use 
         stat models as it provides more information on the model (modules are better)
"""

import numpy as np
import common_functions


def check_for_high_correl_between_OrigVariables(df,column_names):
    """check if there is high correlation between variables"""
    for c in xrange(len(column_names)-1):
        for cn in xrange(c+1,len(column_names)):
            corr = common_functions.determine_correlation(df[column_names[c]],df[column_names[cn]])
            if abs(corr) >.76:
                print "var",column_names[c],column_names[cn],"are correlated, drop one of the vars"


def def_cross_validation_subsets(df,numK=5):
    df['train'] = -1
    for i in xrange(len(df.index)):
        df['train'][i] = i%numK
    return df

def var_transform_root(df,var_name):
    col = np.array(df[var_name])
    if np.sum(np.isnan(col))>0: print var_name,"orig contain nans"
    if col.min() >= 0:
        col_sqrt = np.sqrt(col)
        df[var_name+"_Root2"] = col_sqrt
    elif col.max() <= 0:
        col_sqrt = np.sqrt(-col)
        df[var_name+"_NegRoot2"] = col_sqrt
    else:
        col_sqrt = common_functions.silly_cuberoot(col)
        df[var_name+"_Root3"] = col_sqrt
    return df
    
def var_transform_log(df,var_name):
    col = np.array(df[var_name])
    if col.min() >= 0:
        col_sqrt = np.log1p(col)
        df[var_name+"_Log"] = col_sqrt
    elif col.max() <= 0:
        col_sqrt = np.log1p(-col)
        df[var_name+"_NegLog"] = col_sqrt
    return df

def var_transform_inverse(df,var_name):
    new_var_name = var_name+"_inv"
    df[new_var_name] = 0
    for i in xrange(len(df.index)):
        if df[var_name][i] != 0:
            df [new_var_name][i] = 1.0/float(df[var_name][i])
    return df
            
            
def var_transform_interaction(df,var1,var2):
    c1 = np.array(df[var1])
    c2 = np.array(df[var2])
    df[var1+"_X_"+var2] = np.multiply(c1,c2)
    return df

def var_transform_range(df,var_name):
    vec = np.array(df[var_name])
    min_ = np.min(vec)
    max_ = np.max(vec)
    if min_ != max_:
        df[var_name+"_range01"] = (vec-min_)/float((max_-min_))
        return df
    else:
        return df

