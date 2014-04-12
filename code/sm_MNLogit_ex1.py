# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:32:42 2014

@author: ogavril

PURPOSE: this is an example of a MULTINOMIAL Logistic regression;
         It's based on test06, but the response variable is PID
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm 

import DF_functions
import model_functions
import plot_functions
import common_functions


def var_column_names(df):
    """this function must be changed per each DF used
        This function returns the column names which are the variables to be used for fitting"""
    var_names = []
    for col in df.columns:
        if col not in ["train","const","PID", 'logpopul']:
            var_names.append(col)
    return var_names

        
  
    
"""getting data"""   
fn = ".."+os.sep+"data"+os.sep +"test06.csv"
df = pd.read_csv(fn)

print "column names",df.columns
print "number of observations:",len(df.index)

column_names = ['popul', 'TVnews', 'selfLR', 'ClinLR', 'DoleLR', 'age', 'educ', 'income', 'vote',]
df = df.dropna()
print "number of observations (after dropping NaNs):",len(df.index)

DF_functions.check_for_high_correl_between_OrigVariables(df,column_names)



df = DF_functions.def_cross_validation_subsets(df,numK=5)
var_names = var_column_names(df)


df["const"] = 1
yVar = 'PID'
yVarN = [yVar]
xVarN = var_names+["const"]


mnlog = sm.MNLogit(df[yVar],df[xVarN]).fit()
print mnlog.summary()
print "parameters of MNLogit\n",mnlog.params
#
#
#log_model = model_functions.Logit_iteration(df,yVarN,xVarN,pltname="test06_OrigVarVOTE")
#sensitivity,specificity,precision = model_functions.calculate_SensSpecifPrec(df,log_model,yVar,TH=0.5)
#print "For the complete set of observations sensitivity %f\n specificity %f\n precision %f\n" %(sensitivity,specificity,precision)
##plot_functions.plot_xVar_yVar_logit(df,'PID','pred',title='')
#
#
#
#"""transforming vars"""
#cols2transform = ['popul','age']
#for col_name in cols2transform:
#    df = DF_functions.var_transform_log(df,col_name)
#    df = DF_functions.var_transform_range(df,col_name)
#    
#var_names = var_column_names(df)
#xVarN = var_names+["const"]
#xVarN.remove('age')
#xVarN.remove('popul')
#
#print "\nadding transformed vars"
#log_model = model_functions.Logit_iteration(df,yVarN,xVarN,pltname="test06_OrigVar+")
#sensitivity,specificity,precision = model_functions.calculate_SensSpecifPrec(df,log_model,yVar,TH=0.5)
#print "For the complete set of observations sensitivity %f\n specificity %f\n precision %f\n" %(sensitivity,specificity,precision)
#
#signif_xVarN = xVarN
#for sig in [0.3,0.1]:
#    signif_xVarN = [var for var in signif_xVarN if log_model.pvalues[var] <= sig]   
#    log_model = model_functions.Logit_iteration(df,yVarN,signif_xVarN,pltname="test06_"+str(sig))
#print "\nonly SELECTED & SIGNIFICANT variables are let in the model"
#sensitivity,specificity,precision = model_functions.calculate_SensSpecifPrec(df,log_model,yVar,TH=0.5)
#print "For the complete set of observations sensitivity %f\n specificity %f\n precision %f\n" %(sensitivity,specificity,precision)
#
#
#cross_validation(df)