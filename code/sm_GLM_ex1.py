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

def var_column_names(df):
    """this function must be changed per each DF used
        This function returns the column names which are the variables to be used for fitting"""
    var_names = []
    for col in df.columns:
        if col not in ['NABOVE','NBELOW',"train","const"]:
            var_names.append(col)
    return var_names
    
def applying_OLS(df,yVarN,xVarN):
    lin_model = model_functions.OLS_iteration(df,yVarN,xVarN,pltname='test03_LinRegAllVar')
    
    signif_xVarN = []
    for var in xVarN:
        lin_model = model_functions.OLS_iteration(df,yVarN,[var],pltname=None)
        if lin_model.pvalues[var]<= 0.1: signif_xVarN.append(var)
     
    
    lin_model = model_functions.OLS_iteration(df,yVarN,signif_xVarN,pltname='test03_sigVarOnly')
    
    #signif_xVarN = xVarN
    for sig in [0.3,0.1,0.05]:
        signif_xVarN = [var for var in signif_xVarN if lin_model.pvalues[var] <= sig]
        lin_model = model_functions.OLS_iteration(df,yVarN,signif_xVarN,pltname='test03_LinReg_VarP'+str(sig))
    
"""getting data"""   
fn = ".."+os.sep+"data"+os.sep +"test03.csv"
df = pd.read_csv(fn)

print "column names",df.columns
print "number of observations:",len(df.index)

column_names = ['LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP', 'PERMINTE', 'AVYRSEXP', 'AVSALK', 'PERSPENK', 'PTRATIO',
                'PCTAF', 'PCTCHRT', 'PCTYRRND', 'PERMINTE_AVYRSEXP', 'PERMINTE_AVSAL', 'AVYRSEXP_AVSAL', 
                'PERSPEN_PTRATIO', 'PERSPEN_PCTAF', 'PTRATIO_PCTAF', 'PERMINTE_AVYRSEXP_AVSAL', 'PERSPEN_PTRATIO_PCTAF']
df = df.dropna()
print "number of observations:",len(df.index)

df = DF_functions.def_cross_validation_subsets(df,numK=5)
var_names = var_column_names(df)

"""trying OLS without any var transformation"""
df["const"] = 1

yVarN = ['NABOVE']#, 'NBELOW']
xVarN = var_names+["const"]

#glm_model = model_functions.GLM_iteration(df,yVarN,xVarN,pltname="test03_GLS_OrigVars") 
glm_model = sm.GLM(df[yVarN],df[xVarN],family=sm.families.Gaussian() ).fit()
print "\nGLM using",len(xVarN),"variables"
print glm_model.summary()
print "model's AIC:",glm_model.aic
residuals = glm_model.resid_response
y_pred = glm_model.predict()
plot_functions.plots2_TrueVSpred_Resid(df[yVarN],y_pred,residuals=residuals,weights=None,y_true=None,figfn ="GLM_OrigVars")




"""transforming vars"""
for col_name in column_names:
    df = DF_functions.var_transform_root(df,col_name)
    df = DF_functions.var_transform_log(df,col_name)
    df = DF_functions.var_transform_interaction(df,col_name,col_name)

for c in xrange(len(column_names)-1):
    for cn in xrange(c+1,len(column_names)):
        df = DF_functions.var_transform_interaction(df,column_names[c],column_names[cn])
var_names = var_column_names(df)

yVarN = ['NABOVE']#, 'NBELOW']
xVarN = var_names+["const"]

glm_model = model_functions.GLM_iteration(df,yVarN,xVarN,pltname="test03_GLS_AllVars") 

y2VarN = ['NABOVE','NBELOW'] 
glm2_model = sm.GLM(df[y2VarN],df[xVarN],family=sm.families.Binomial()).fit()
print "\nGLM using",len(xVarN),"variables"
print glm2_model.summary()
print "model's AIC:",glm2_model.aic
residuals = glm2_model.resid_response
y_pred = glm2_model.predict()


    
"""things to try when everything seems wrong
    1) look for missing variables, interations, transformations
    2) outliering
    
"""