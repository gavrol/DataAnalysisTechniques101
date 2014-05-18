# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 08:53:58 2014

@author: ogavril

purpose:  testing Main on test06
"""

import os
import pandas as pd
import numpy as np
import  statsmodels.api as sm 
import modelling_functions
import data_functions
import common_functions


####test06 stuff starts here    
def var_column_names(df):
    """this function must be changed per each DF used
        This function returns the column names which are the variables to be used for fitting"""
    var_names = []
    for col in df.columns:
        if col not in ["train","const",'PID', 'logpopul','_TRAIN']:
            var_names.append(col)
    return var_names

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

def evaluate_MNLogit(predicted_prob_mtx,responses):
    predicted_values = np.array([0 for i in range(predicted_prob_mtx.shape[0])])
    for r in range(predicted_prob_mtx.shape[0]):
        max_v = max(predicted_prob_mtx[r])
        max_c = 0        
        for c in range(predicted_prob_mtx.shape[1]):
            if predicted_prob_mtx[r][c] >= max_v:
                max_c = c
                break
        predicted_values[r] = responses[max_c]
    return predicted_values
        
            
    
"""getting data"""   
fn = ".."+os.sep+"data"+os.sep +"test06.csv"
df = pd.read_csv(fn)

print "column names",df.columns
print "number of observations:",len(df.index)

column_names = ['popul', 'TVnews', 'selfLR', 'ClinLR', 'DoleLR', 'age', 'educ', 'income', 'vote']
df = df.dropna()
print "number of observations:",len(df.index)



train_var = '_TRAIN'
df = common_functions.def_cross_validation_subsets(df,train_var,numK=7)
trs_npArray = np.array(df[train_var])

target_name = 'PID'
random_prediction_target = data_functions.randomize_prediction_v1(df,target_name)
print target_name

"""transforming vars"""
cols2transform = ['popul']#,'age']
for col_name in cols2transform:
    df = var_transform_log(df,col_name)
    #df = DF_functions.var_transform_range(df,col_name)
#cols2transform = ['age']
#for col_name in cols2transform:
#    #df = DF_functions.var_transform_log(df,col_name)
#    df = DF_functions.var_transform_range(df,col_name)
    
df["const"] = 1    
var_names = var_column_names(df)
xVarN = var_names+["const"]
   
xVarN.remove('age')
xVarN.remove('popul')
xVarN.remove('income')


##### starting model choice #######

MODEL_STATSD = {} #dictionary to fill with model
MODELS = []
MODEL_NAMES = ["MNLogit"]#,"SVM","GBT"]#,"Logit",]#'SVM',"LOGLOG","AdaBoost","FOREST","NaiveBayes",] 


model_pass = {'FOREST':False,'AdaBoost':False,'GBT':False,"SVM":False}#to enble pass for some models if they have been ran once
TrainModel = {}
validation_set = 4

for MODEL_NAME in MODEL_NAMES:
    Normalize_Vars = True
    
    #xVarN = var_column_names(df)
    print "\n\n applying",MODEL_NAME   
        
    if MODEL_NAME in ["MNLogit","LOGLOG"] and ('const' not in df.columns):
        df["const"] = 1
        xVarN  += ["const"]
    print "starting independent variables",xVarN
    
    
    TrainModel_Stats = {}
    TestModel_Stats = {}


    train_sets = range(4)#[elem for elem in df[train_var].unique() if elem != validataion_set]
       
        
    for trs in train_sets:
        
        print "\n TEST set:",trs,"(i.e., train set excludes set",trs,'and',validation_set,")"
        if MODEL_NAME in ['FOREST','AdaBoost','GBT',"SVM","MNLogit"]:
            TRAIN_df = df[(df['_TRAIN'] != validation_set)]
        else:
            TRAIN_df = df[(df['_TRAIN']!=trs) & (df['_TRAIN'] != validation_set)]
        TEST_df = df[df['_TRAIN']==trs]
        TrainModel_Stats[trs] = {}
        TestModel_Stats[trs] = {}
    
        if MODEL_NAME == "MNLogit": 
            print MODEL_NAME,"for testset",trs
            #kernels = modelling_functions.use_MNLogit(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,Normalize_Vars)
            train_df = TRAIN_df.copy()
            test_df = TEST_df.copy()
            train_target = np.array(train_df[target_name])
            test_target = np.array(test_df[target_name])
            if 'const' not in train_df.columns:
                train_df['const'] = 1
            if 'const' not in test_df.columns:
                test_df['const'] = 1
        
            log_model = sm.MNLogit(train_df[target_name],train_df[xVarN]).fit()
            print log_model.summary()
            predicted_values_train = evaluate_MNLogit(log_model.predict(),[0,1,2,3,4,5,6]) 
            #print "predicted_values_train", predicted_values_train
            tr_correct_predictions = modelling_functions.correctly_predicted(predicted_values_train,train_target)
            print "correctly predicted:",tr_correct_predictions
            AdH_predicted_values_validation = random_prediction_target[trs_npArray!=validation_set]
            AdH_accuracy =  modelling_functions.correctly_predicted(AdH_predicted_values_validation,train_df['PID'][train_df[train_var] != validation_set])                  

            print "correctly predicted by AdH_predicted_values_validation", AdH_accuracy
            
