# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:34:56 2014

@author: ogavril
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

#"""transforming vars"""
#cols2transform = ['popul','age']
#for col_name in cols2transform:
#    df = DF_functions.var_transform_log(df,col_name)
#    df = DF_functions.var_transform_range(df,col_name)

target_name = 'vote'
xVarN = var_column_names(df)
#xVarN.remove('age')
#xVarN.remove('popul')

df = DF_functions.def_cross_validation_subsets(df,numK=5)
TrainModel_Stats = {}
TestModel_Stats = {}
train_sets = df['train'].unique()
for trs in [0]:#train_sets:

    print "\n TEST set:",trs,"(i.e., train set excludes set",trs,")"
    TRAIN_df = df[df['train']!=trs]
    TEST_df = df[df['train']==trs]
    TrainModel_Stats[trs] = {}
    TestModel_Stats[trs] = {}

    
    for kernel in ['linear','rbf','sigmoid','poly',]:
        print "\n kernel =",kernel
        train_df = TRAIN_df.copy()
        test_df = TEST_df.copy()
        train_data,train_target,test_data,test_target = skML_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=True)

        predicted_values_train,predicted_values_test = skML_functions.perform_svm(train_data,train_target,test_data,test_target,kernel,polydeg=3)
        
        train_df['pred'] = predicted_values_train
        sensitivity,specificity,precision,accuracy = model_functions.SensSpecifPrec(train_df,'vote',TH=0.99)
        print "For the train set of observations sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)
        TrainModel_Stats[trs][kernel] ={'sensitivity':sensitivity,"specificity":specificity,'precision':precision,
                                    'accuracy': accuracy}  
#        Nsensitivity,Nspecificity,Nprecision,Naccuracy = model_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
#        print sensitivity-Nsensitivity,specificity-Nspecificity,precision-Nprecision,accuracy-Naccuracy

        test_df['pred'] = predicted_values_test                            
        sensitivity,specificity,precision,accuracy = model_functions.SensSpecifPrec(test_df,'vote',TH=0.99)
        print "For the test set of observations sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)
        TestModel_Stats[trs][kernel] ={'sensitivity':sensitivity,"specificity":specificity,'precision':precision,
                                    'accuracy': accuracy}  
#        Nsensitivity,Nspecificity,Nprecision,Naccuracy = model_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
#        print sensitivity-Nsensitivity,specificity-Nspecificity,precision-Nprecision,accuracy-Naccuracy
      

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
