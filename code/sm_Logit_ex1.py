# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:32:42 2014

@author: ogavril

PURPOSE: using statsmodels package NOT scikit
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm #using statsmodels instead of scikit

import DF_functions
import model_functions
import plot_functions
import common_functions


def var_column_names(df):
    """this function must be changed per each DF used
        This function returns the column names which are the variables to be used for fitting"""
    var_names = []
    for col in df.columns:
        if col not in ['GRADE',"train","const"]:
            var_names.append(col)
    return var_names

       
    
"""getting data"""   
fn = ".."+os.sep+"data"+os.sep +"test05_logit.csv"
df = pd.read_csv(fn)

print "column names",df.columns
print "number of observations:",len(df.index)

column_names = ['GPA', 'TUCE', 'PSI', 'GRADE']
df = df.dropna()
print "number of observations:",len(df.index)

df = DF_functions.def_cross_validation_subsets(df,numK=5)
var_names = var_column_names(df)


df["const"] = 1
yVar = "GRADE"
yVarN = [yVar]
xVarN = var_names+["const"]


log_model = model_functions.Logit_iteration(df,yVarN,xVarN,pltname="test05_OrigVar")
sensitivity,specificity,precision,accuracy = model_functions.calculate_SensSpecifPrec(df,log_model,yVar,TH=0.4)
print "For Training set sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)

##"""transforming vars"""
##cols2transform = ['TUCE',]
##for col_name in cols2transform:
##    df = DF_functions.var_transform_log(df,col_name)
##    df = DF_functions.var_transform_range(df,col_name)

#var_names = var_column_names(df)
#xVarN = var_names+["const"]
##xVarN.remove("TUCE")
##xVarN.remove('TUCE_range01')
#print xVarN
#log_model = model_functions.Logit_iteration(df,yVarN,xVarN,pltname="test05_VarSet1")
#sensitivity,specificity,precision,accuracy = model_functions.calculate_SensSpecifPrec(df,log_model,yVar,TH=0.5)
#print "For Training set sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)


signif_xVarN = xVarN
for sig in [0.1]:
    signif_xVarN = [var for var in signif_xVarN if log_model.pvalues[var] <= sig]   
    log_model = model_functions.Logit_iteration(df,yVarN,signif_xVarN,pltname="test05_"+str(sig))
#
plot_functions.plot_xVar_yVar_logit(df,'GPA',yVar,title='GPAvsImprovement',figname='GPAvsImprovement',model=log_model)
sensitivity,specificity,precision,accuracy = model_functions.calculate_SensSpecifPrec(df,log_model,yVar,TH=0.5)
print "For Training set sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)



model_functions.evaluate_model_effectiveness_logit(df,yVar,log_model,TH=0.5,is_test_set=False)

#
##"""let's try with GLM"""
##glm_model = sm.GLM(df[yVarN],df[xVarN],family=sm.families.Gaussian(sm.families.links.log) ).fit()
##print "\nGLM using",len(xVarN),"variables"
##print glm_model.summary()
##residuals = glm_model.resid_response
##y_pred = glm_model.predict()
##plot_functions.plots2_TrueVSpred_Resid(df[yVarN],y_pred,residuals=residuals,weights=None,y_true=None,figfn ="GLM_OrigVars")
