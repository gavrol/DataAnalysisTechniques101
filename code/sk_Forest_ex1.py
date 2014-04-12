# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:34:56 2014

@author: ogavril

PURPOSE: how do CARTs perform to predict voting (test06)??

CONCLUSION: 1) CARTs seem to be fast and no variable transformation is needed (so similar to SVM)
            2) accuracy with CARTs is worse than with logistic reg (but no var transformation is needed)
"""

import os
import numpy as np
import pandas as pd

import DF_functions
import model_functions
import skML_functions
from class_definitions import *
 
def var_column_names(df):
    """this function must be changed per each DF used
        This function returns the column names which are the variables to be used for fitting"""
    var_names = []
    for col in df.columns:
        if col not in ["train","const",'vote', 'logpopul']:
            var_names.append(col)
    return var_names

  
"""getting data"""   
fn = ".."+os.sep+"data"+os.sep +"test06.csv"
df = pd.read_csv(fn)

print "column names",df.columns
print "number of observations:",len(df.index)

column_names = ['popul', 'TVnews', 'selfLR', 'ClinLR', 'DoleLR', 'age', 'educ', 'income', 'PID',]
df = df.dropna()
print "number of observations:",len(df.index)

DF_functions.check_for_high_correl_between_OrigVariables(df,column_names)


target_name = 'vote'
xVarN = var_column_names(df)
print "variables included in the CART procedure", xVarN

df = DF_functions.def_cross_validation_subsets(df,numK=5)
TrainModel_Stats = {}
TestModel_Stats = {}
train_sets = df['train'].unique()
for trs in train_sets:

    print "\n TEST set:",trs,"(i.e., train set excludes set",trs,")"
    TRAIN_df = df[df['train']!=trs].copy()
    TEST_df = df[df['train']==trs].copy()

    TrainModel_Stats[trs] = {}
    TestModel_Stats[trs] = {}

    for max_depth in [1,2,3,4,5,6]:
        for max_features in [1,2,3,4,5,6]:
            train_df = TRAIN_df.copy()
            test_df = TEST_df.copy()
            kernel = 'md_'+str(max_depth)+"_mf_"+str(max_features)
            
            train_data,train_target,test_data,test_target = skML_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=True)
            predicted_values_train,predicted_values_test = skML_functions.perform_RandomForest( train_data,train_target,test_data,test_target,
                                                                                   num_trees=10,RF_max_depth=max_depth, RF_max_features=max_features,
                                                                                   RF_random_state=1,classifier='ExtraTree') #'RandomForest')    
           
            train_df['pred'] = predicted_values_train
            sensitivity,specificity,precision,accuracy = model_functions.SensSpecifPrec(train_df,'vote',TH=0.99)
            #print "For the train set of observations sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)
            TrainModel_Stats[trs][kernel] ={'sensitivity':sensitivity,"specificity":specificity,'precision':precision,
                                        'accuracy': accuracy}  
            test_df['pred'] = predicted_values_test                            
            sensitivity,specificity,precision,accuracy = model_functions.SensSpecifPrec(test_df,'vote',TH=0.99)
            #print "For the test set of observations sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)
            TestModel_Stats[trs][kernel]={'sensitivity':sensitivity,"specificity":specificity,'precision':precision,
                                        'accuracy': accuracy}  

models = skML_functions.summarize_model_stats(TestModel_Stats,['sensitivity',"specificity",'precision','accuracy'])
best_avg = 0.0

for model in models:
    print model.name
    avg = 0
    for stat in ['sensitivity',"specificity",'precision','accuracy']:
        print stat,'=', model.__dict__[stat]
        avg += model.__dict__[stat]
    avg = avg/4.0
    if avg/4.0 > best_avg:
        best_model = model.name
        best_avg = avg/4.0
print "\nbest model is:", best_model
for model in models:
    if model.name == best_model:
        for stat in ['sensitivity',"specificity",'precision','accuracy']:
            print stat,'=', model.__dict__[stat]


