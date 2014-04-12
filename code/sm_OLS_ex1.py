# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:32:42 2014

@author: ogavril

PURPOSE: using statsmodels package NOT scikit
"""
import os
import numpy as np
import pandas as pd

import DF_functions
import model_functions


def var_column_names(df):
    """this function must be changed per each DF used
        This function returns the column names which are the variables to be used for fitting"""
    var_names = []
    for col in df.columns:
        if col not in ["response","train","const"]:
            var_names.append(col)
    return var_names
    
#def OLS_iteration(df,yVarN,xVarN,pltname='plot'):
#    lin_model = sm.OLS(df[yVarN],df[xVarN]).fit()
#    print lin_model.summary()
#    residuals = lin_model.resid
#    y_pred = lin_model.predict()
#    plots.plots2_TrueVSpred_Resid(df[yVarN],y_pred,residuals=residuals,weights=None,y_true=None,figfn =pltname)
#    return lin_model
    
fn = ".."+os.sep+"data"+os.sep +"test04.csv"
df = pd.DataFrame.from_csv(fn)
print "column names",df.columns
print "number of observations:",len(df.index)

column_names = ['f1', 'f2', 'f3', 'f4', 'f5','f6']
df = df.dropna()
df = DF_functions.def_cross_validation_subsets(df,numK=5)


"""running the Lin model just with the original variables"""
var_names = var_column_names(df)
df["const"] = 1
yVarN = ["response"]
xVarN = var_names+["const"]
lin_model = model_functions.OLS_iteration(df,yVarN,xVarN,pltname='LinReg_origVar')


"""transforming variables, adding interactions"""
for col_name in column_names:
    df = DF_functions.var_transform_root(df,col_name)
    df = DF_functions.var_transform_log(df,col_name)
    df = DF_functions.var_transform_interaction(df,col_name,col_name)

for c in xrange(len(column_names)-1):
    for cn in xrange(c+1,len(column_names)):
        df = DF_functions.var_transform_interaction(df,column_names[c],column_names[cn])

var_names = var_column_names(df)
xVarN = var_names+["const"]
    

lin_model = model_functions.OLS_iteration(df,yVarN,xVarN,pltname='LinRegAllVar')
MSE = model_functions.evaluate_SMmodel_on_testSet(lin_model,df,yVarN)
print "MSE =",MSE, "\n"

   
    
#signif_xVarN = [var for var in xVarN if lin_model.pvalues[var] <= 0.5]
#
#lin_model = model_functions.OLS_iteration(tdf,yVarN,signif_xVarN,pltname='LinReg_VarP05')
#MSE = model_functions.evaluate_SMmodel_on_testSet(lin_model,testDF,yVarN)
#print MSE, "\n"
#signif_xVarN = [var for var in signif_xVarN if lin_model.pvalues[var] <= 0.1]
#
#lin_model = model_functions.OLS_iteration(tdf,yVarN,signif_xVarN,pltname='LinReg_VarP005')
#MSE = model_functions.evaluate_SMmodel_on_testSet(lin_model,testDF,yVarN)
#print MSE, "\n"


#for test_set in xrange(5):
#    print "\n\n TEST Set",test_set
#    tdf = df[df['train']!= test_set]
#    testDF = df[df['train'] == test_set]
#    
#    var_names = var_column_names(tdf)
#    xVarN = var_names+["const"]
#    
#    lin_model = model_functions.OLS_iteration(tdf,yVarN,xVarN,pltname='LinRegAllVar')
#    MSE = model_functions.evaluate_SMmodel_on_testSet(lin_model,testDF,yVarN)
#    print "MSE =",MSE, "\n"
#    
#    
#    signif_xVarN = xVarN
#    for sig in [0.5,0.2,0.05]:
#        signif_xVarN = [var for var in signif_xVarN if lin_model.pvalues[var] <= sig]   
#        lin_model = model_functions.OLS_iteration(tdf,yVarN,signif_xVarN,pltname='LinReg_VarP'+str(sig))
#        MSE = model_functions.evaluate_SMmodel_on_testSet(lin_model,testDF,yVarN)
#        print "MSE =",MSE, "\n"
        
    
"""things to try when everything seems wrong
    0) filling NaNs
    1) look for missing variables, interations, transformations
    2) outliering
    
"""