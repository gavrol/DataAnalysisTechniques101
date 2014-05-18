# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:57:28 2014

@author: ogavril

PURPOSE: functions called by FHB_main.py and FHBagg_main.py

OUTPUT: none
"""
import os 
import sys
import numpy as np

from class_definitions import *

from sklearn import metrics,ensemble
from sklearn import grid_search,svm
import statsmodels.api as sm
sys.path.append(os.path.normpath(os.path.join(sys.path[0],".."+os.sep+".."+os.sep+'common_py'+os.sep))) 
import plot_functions
import common_functions

def perform_CART(train_data,train_target,test_data,test_target,min_samples_split_,min_samples_leaf_, DT_max_features=None,DT_random_state=None):
    mod = tree.DecisionTreeClassifier(min_samples_split=min_samples_split_,min_samples_leaf=min_samples_leaf_, max_features=DT_max_features,random_state=DT_random_state,n_jobs=-1)
    #print mod
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod


def perform_AdaBoost(train_data,train_target,test_data,test_target,min_samples_split_,min_samples_leaf_, DT_max_features=None,DT_random_state=None,num_estimators=50):
    mod = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(compute_importances=None, criterion='gini',
                                                        min_samples_split =min_samples_split_,min_samples_leaf=min_samples_leaf_,
                                                         max_features=DT_max_features,random_state=DT_random_state,splitter='best'), 
                                                         n_estimators=num_estimators, learning_rate=0.05, algorithm='SAMME.R')

    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod



def use_AdaBoost(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,Features,NumEstimators,min_samples_split,min_samples_leaf):
    kernels = []

    for max_features in Features: #xrange(10,250,10): 
        for num_estimator in NumEstimators:
            train_df = TRAIN_df.copy()
            test_df = TEST_df.copy()
            kernel = 'AdaBoost_mF_'+str(max_features)+"_nE_"+str(num_estimator)
            kernels.append(kernel)
            if kernel not in TrainModel.keys():
                TrainModel[kernel] = {}

            train_data,train_target,test_data,test_target = common_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)

            predicted_values_train,predicted_values_test,model = perform_AdaBoost(train_data,train_target,test_data,test_target,min_samples_split,min_samples_leaf,
                                                                                num_estimators=num_estimator, DT_max_features=max_features,
                                                                                   DT_random_state=None)
       
            TrainModel[kernel][trs] = {}
            TrainModel[kernel][trs]['model'] = model               
    
#            tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
#            #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
#            TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
#            TestModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  

            """TESTING is not needed for trees                             
            ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
            #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
            TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}
            """ 
    return kernels




def perform_GBT(train_data,train_target,test_data,test_target,min_samples_split_,min_samples_leaf_,max_depth=1,learning_rate =0.1,num_estimators=100,max_features ='log2'):
    """Gradient Boosting Tree"""                  
    mod = ensemble.GradientBoostingClassifier(n_estimators=num_estimators, learning_rate=learning_rate, min_samples_split=min_samples_split_,min_samples_leaf=min_samples_leaf_,
                                              max_depth=max_depth, max_features = max_features,subsample=0.6,random_state=1357,loss='deviance')
                                              #learning_rate=0.01, n_estimators=3000, min_samples_split=12, min_samples_leaf=12   max_depth=6, 
    #print mod
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod



def use_GBT(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,NumInteractions,NumBoostingStages,Features,min_samples_split,min_samples_leaf,learning_rate_):
    kernels = []
    for max_num_inter in NumInteractions: #xrange(10,350,20):
        for num_estimator in NumBoostingStages: #xrange(50,250,50)
            for feature in Features:
                train_df = TRAIN_df.copy()
                test_df = TEST_df.copy()
                kernel = 'GBT_mI_'+str(max_num_inter)+"_mE_"+str(num_estimator)+"_F_"+str(feature)
                #print kernel
                kernels.append(kernel)
                if kernel not in TrainModel.keys():
                    TrainModel[kernel] = {}
    
                train_data,train_target,test_data,test_target = common_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)
                
                predicted_values_train,predicted_values_test,model = perform_GBT(train_data,train_target,test_data,test_target,min_samples_split,min_samples_leaf,
                                                                                                     max_depth=max_num_inter,learning_rate =learning_rate_,
                                                                                                     num_estimators=num_estimator,max_features =feature)
                TrainModel[kernel][trs] = {}
                TrainModel[kernel][trs]['model'] = model
        
                tr_correct_predictions = correctly_predicted(predicted_values_train,train_target)
                TestModel_Stats[trs][kernel] ={'accuracy':tr_correct_predictions}
                #print 'accuracy = ',tr_correct_predictions
    return kernels
    
def use_Forest(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,Features,NumTrees,min_samples_split,min_samples_leaf):
    kernels = []

    for max_features in Features: #xrange(10,250,10): 
        for num_trees in NumTrees:
            train_df = TRAIN_df.copy()
            test_df = TEST_df.copy()
            kernel = 'Forest_mF_'+str(max_features)+"_nT_"+str(num_trees)
            
            kernels.append(kernel)
            if kernel not in TrainModel.keys():
                TrainModel[kernel] = {}

            train_data,train_target,test_data,test_target = common_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)

            predicted_values_train,predicted_values_test,model = perform_RandomForest(train_data,train_target,test_data,test_target,
                                                                                                        min_samples_split,min_samples_leaf, num_trees=num_trees, 
                                                                                                        RF_max_features=max_features,RF_random_state=1,classifier='RandomForest')
        
            TrainModel[kernel][trs] = {}
            TrainModel[kernel][trs]['model'] = model               
    
            tr_correct_predictions = correctly_predicted(predicted_values_train,train_target)
            TestModel_Stats[trs][kernel] ={'accuracy':tr_correct_predictions}
    return kernels
def perform_RandomForest(train_data,train_target,test_data,test_target,min_samples_split_,min_samples_leaf_, num_trees=10, RF_max_depth=None, RF_max_features=None,RF_random_state=None,classifier='RandomForest'):
    if classifier.lower()== 'randomforest':    
        mod = ensemble.RandomForestClassifier(n_estimators = num_trees,max_depth=RF_max_depth, max_features=RF_max_features,random_state=RF_random_state,
                                              min_samples_split=min_samples_split_,min_samples_leaf=min_samples_leaf_,n_jobs=-1)
    else:
        mod = ensemble.ExtraTreesClassifier(n_estimators = num_trees,max_depth=RF_max_depth, max_features=RF_max_features,random_state=RF_random_state,
                                            min_samples_split =min_samples_split_,min_samples_leaf=min_samples_leaf_,n_jobs=-1)
    #print mod n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod



def use_SVM(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,kernels,poly_degrees):
    train_df = TRAIN_df.copy()
    test_df = TEST_df.copy()
    train_data,train_target,test_data,test_target = common_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)

    for kernel in kernels:
        if kernel == 'rbf':
            parameters = {'kernel': [kernel], 'C': [1, 10,100] , 'gamma': [1e-3, 1e-1, 1.0]} #'C': [1, 10,100]  'gamma': [1e-3, 1e-1, 1.0],
        elif kernel == 'linear':
            parameters = {'kernel': [kernel], 'C': [1, 10,100]  } #'C':[1, 10, 100]
        elif kernel in [ 'poly','sigmoid']:
            parameters = {'kernel': [kernel], 'degree':poly_degrees, 'C': [10], 'gamma': [1e-3, 1e-1, 1.0]}
        else:
            print "!!! unknown SVM kernel"
            return None
            
        kernel = "SVM_"+kernel
        print kernel
        if kernel not in TrainModel.keys():
            TrainModel[kernel] = {}
        classifier = svm.SVC()
        mod = grid_search.GridSearchCV(classifier, parameters) #, scoring=metrics.f1_score) #score_func)
        mod.fit(train_data, train_target, cv=3)
        predicted_values_train = mod.predict(train_data)
        print "Best parameters set found on development set: \n", mod.best_estimator_
        predicted_values_test = mod.predict(test_data)
#        print metrics.classification_report(test_target, predicted_values_test)

        TrainModel[kernel][trs] = {}
        TrainModel[kernel][trs]['model'] = mod
    
        tr_correct_predictions = correctly_predicted(predicted_values_train,train_target)
        TestModel_Stats[trs][kernel] ={'accuracy':tr_correct_predictions}
    return kernels

def use_MNLogit(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE):
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
    predicted_values_train = log_model.predict() #train_df[xVarN])
    print "predicted_values_train", predicted_values_train
    predicted_values_test  = log_model.predict(test_df[xVarN])


    kernel = "MNLogit"
    if kernel not in TrainModel.keys():
        TrainModel[kernel] = {}

    TrainModel[kernel][trs] = {}
    TrainModel[kernel][trs]['model'] = log_model               

    tr_correct_predictions = correctly_predicted(predicted_values_train,train_target)
    TestModel_Stats[trs][kernel] ={'accuracy':tr_correct_predictions}

    return [kernel]


def correctly_predicted(predicted_values_train,train_target):
    a = np.abs(predicted_values_train - train_target)
    count_zeros = len(a) - np.count_nonzero(a)
    return float(count_zeros)/float(len(a))
    
      
####### model summaries ###########
       

 
       
def summarize_model_performance(TestModel_Stats,stats,logF=None):
    """because each model is run on a few validaton/test sets, 
    the function averages the stats for each model based on which test/validation set was used;
    later one should run determine_best_model()  
    There are model types for which there is only one train set, e.g., Forest, but that's not important
    """
    train_sets = TestModel_Stats.keys()
    kernels = []
    for trs in TestModel_Stats.keys():
        for kernel in  TestModel_Stats[trs].keys():
            if kernel not in kernels:
                kernels.append(kernel)
    models = []
        
    for kernel in kernels:
        model = DISCRETE_MODEL(kernel)
        for stat in stats:
            t,count = 0,0
            for trs in train_sets:
                if kernel in TestModel_Stats[trs].keys():
                    t += TestModel_Stats[trs][kernel][stat]
                    count += 1
            model.__dict__[stat] = t/float(count)
        models.append(model)
        
#    if logF != None:
#        for model in models:
#            logF.write(model.name+"\n")
#            for stat in stats:
#                logF.write(stat+":"+str(round(model.__dict__[stat],4))+"\n")
    return models
          



def determine_best_model(models,stats=None,stat_weights=None,logF=None):
    """to run AFTER summarize_model_performance() and after evaluating model on validation set"""
    best_avg = 0.0
    best_model_name = ""
    if stats == None:
        stats = ['sensitivity',"specificity",'precision','accuracy']

    if stat_weights == None:
        stat_weights = [1 for stat in stats]
    
    for model in models:
        #print model.name
#        avg = 2*(model.__dict__['sensitivity'] *model.__dict__['precision'])
#        if avg != 0:
#            avg = avg/(model.__dict__['sensitivity'] + model.__dict__['precision'] ) 
        avg = 0        
        for stat in stats: #:
            #print stat,'=', model.__dict__[stat]
            avg += model.__dict__[stat]*stat_weights[stats.index(stat)]
        avg = avg/float(len(stats))
        if avg >= best_avg:
            best_model_name = model.name
            best_avg = avg
    print "\nbest model is:", best_model_name

    for model in models:
        if model.name == best_model_name:
            for stat in stats:#['sensitivity',"specificity",'precision','accuracy']:
                print stat,'=', model.__dict__[stat]
                
    if logF != None:
        logF.write("\nbest model of this type:")
        for model in models:
            if model.name == best_model_name:
                logF.write(best_model_name +"\n")
                for stat in stats:
                    logF.write(stat+":"+str(round(model.__dict__[stat],4))+"\n")
        
    return best_model_name

def evaluate_model(trs_npArray,random_prediction_target,df,validation_set,target_name,TrainModel,best_model,MODEL_NAME,NORMALIZE,xVarN,modelCoeff2file=False,logF=None):

    if validation_set == -1:
        print "\n TESTING on the ENTIRE set"
        VALIDATION_df = df
        AdH_predicted_values_validation = random_prediction_target
    else:
        print "\n VALIDATING",MODEL_NAME,"on validation set (",validation_set,")"
        VALIDATION_df = df[df['_TRAIN']==validation_set]
        AdH_predicted_values_validation = random_prediction_target[trs_npArray==validation_set]

    validation_df = VALIDATION_df.copy()
    validation_target = np.array(validation_df[target_name])
    
    model = None
    for trs in TrainModel[best_model].keys():
        model = TrainModel[best_model][trs]['model']
        break
 
    if MODEL_NAME in ["FOREST", "SVM","NaiveBayes",'GBT','AdaBoost']:
        validation_data = common_functions.make_dataMatrix_fromDF(xVarN,validation_df,normalizeInput=NORMALIZE)    
        predicted_values_validation = model.predict(validation_data)



    AdH_accuracy =  correctly_predicted(AdH_predicted_values_validation,validation_target)                  
    s =  "For AdHoc prediction on validation set of observations \n accuracy %f\n" %(AdH_accuracy)

    vs_accuracy = correctly_predicted(predicted_values_validation,validation_target)                  
    s +=  "For Validation prediction of "+MODEL_NAME+" \n accuracy %f\n" %(vs_accuracy)


    s += "model's correctness improvement: "+str(round(vs_accuracy/AdH_accuracy,2)) +"\n"
    print s
    if logF != None:
        logF.write(s)
               
def predict_on_testSet(df,TrainModel,best_model,MODEL_NAME,NORMALIZE,xVarN,target_name):

    validation_df = df.copy()
    
    model = None
    for trs in TrainModel[best_model].keys():
        model = TrainModel[best_model][trs]['model']
        break
 
    if MODEL_NAME in ["FOREST", "SVM","NaiveBayes",'GBT','AdaBoost']:
        validation_data = common_functions.make_dataMatrix_fromDF(xVarN,validation_df,normalizeInput=NORMALIZE)    
        predicted_values_validation = model.predict(validation_data)
        df['_PREDICTED:'+MODEL_NAME+":"+target_name] = predicted_values_validation
        
    else:
        print "!!! ERROR:",MODEL_NAME,"is not accounted for"
        



    
def evaluate_models_on_validationSet(df,validation_set,target_name,models,TrainModel,MODEL_NAME,NORMALIZE,xVarN,stats=None):
    print "\n EVALUATING models on validation set"
    if stats == None:
        stats = ['accuracy']

    VALIDATION_df = df[df['_TRAIN']==validation_set]
    validation_df = VALIDATION_df.copy()
    validation_target = np.array(validation_df[target_name])
    
    for modObj in models:
        model_name = modObj.name

        model = None
        all_train_sets = [key for key in TrainModel[model_name].keys()]
        trs = all_train_sets[0] #eventually this should be obsolete
        model = TrainModel[model_name][trs]['model']

        if MODEL_NAME in ["FOREST", "SVM","NaiveBayes",'GBT','AdaBoost']:
            validation_data = common_functions.make_dataMatrix_fromDF(xVarN,validation_df,normalizeInput=NORMALIZE)    
            predicted_values_validation = model.predict(validation_data)
    
            accuracy = correctly_predicted(predicted_values_validation,validation_target)
            
        for stat in stats:
            #print stat,"=",eval(stat)
            modObj.__dict__[stat] = modObj.__dict__[stat]*0.0 + eval(stat)*1.0

   
