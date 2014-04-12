# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 07:44:35 2014

@author: ogavril
PURPOSE: general functions to do data analysis of different sorts

"""

import os
import numpy as np
import common_functions
import plots_functions

           
def apply_model(mod,X,y,test_X=None,test_y=None,mod_name=None): 
    mod.fit(X,y)
    y_pred = mod.predict(X)    
    #plots.plot_trueVSpred(y,y_pred,title = mod_name,figname=mod_name)
    np_functions.plots2_TrueVSpred_Resid(mod,y,y_pred,weights=None,y_true=None,figfn =mod_name)
    print mod_name,"coefficients:",mod.coef_
    AE=common_functions.AbsError(y_pred,y)
    print mod_name,"train AbsError=",AE

    if test_X != None and test_y != None:
        y_pred = mod.predict(test_X)
        np_functions.plot_trueVSpred(test_y,y_pred,title = mod_name+" (Test)",figname=mod_name+'_test_')    
        AE=common_functions.AbsError(y_pred,test_y)
        print mod_name,"test AbsError=",AE

def define_simpleX(df,columns):
    X = np.c_[(np.array(df[columns[0]]))]
    
    for c in range(1,len(columns)):
        X = np.c_[(X,np.array(df[columns[c]]))]
    
    return X
    
    
def define_X_with_VarTransformation(df,columns):
    print "Transforming",len(columns),"variables:",columns
    var_names = []
    c = 0
    subX,var_names = variable_transformation(c,df,columns,var_names)
    X = np.c_[(subX)]
    
    for c in range(1,len(columns)):
        subX,var_names = variable_transformation(c,df,columns,var_names)
        X = np.c_[(X,subX)]
    
    return X

  
def variable_transformation(c,df,columns,var_names):
    col =  np.array(df[columns[c]]) 
    var_names.append(columns[c])
    
    colnans = np.isnan(col)
    if np.sum(colnans)>0: print columns[c],"orig contain nans"
    if col.min() >= 0:
        col_sqrt = np.sqrt(col)
        #col1 = np.array(functions_DefaultPred.normalize(col1))
        var_names.append(columns[c]+"_Root2")
    elif col.max() <= 0:
        col_sqrt = np.sqrt(-col)
        #col1 = np.array(functions_DefaultPred.normalize(col1))
        var_names.append(columns[c]+"_NegRoot2")
    else:
        col_sqrt = common_functions.silly_cuberoot(col)
        #col1 = np.array(functions_DefaultPred.normalize(col1))
        var_names.append(columns[c]+"_Root3")
    col1nans = np.isnan(col_sqrt)
    if np.sum(col1nans) >0: print columns[c],"root transf contains nans"


    col_SQ = np.multiply(col,col)
    var_names.append(columns[c]+"_sq")
    
    
    if col.min() > 0:
        col_LOG = np.log(col)
        #col2 = np.array(functions_DefaultPred.normalize(col2))       
        var_names.append(columns[c]+"_Log")
    elif col.max() < 0:
        col_LOG = np.log(-col)
        #col2 = np.array(functions_DefaultPred.normalize(col2))       
        var_names.append(columns[c]+"_NegLog")
    else:
        col_LOG = common_functions.silly_cuberoot(col)
        #col2 = np.array(functions_DefaultPred.normalize(col2))       
        var_names.append(columns[c]+"_Root3")
    col1nans = np.isnan(col_LOG)
    if np.sum(col1nans) >0: print columns[c],"log transf contains nans"
    
    subMat = np.c_[(col,col_SQ,col_sqrt,col_LOG)]
    return subMat,var_names

 