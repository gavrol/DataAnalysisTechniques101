# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 08:27:07 2014

@author: ogavril
PURPOSE: a program to test whether CARTs are deterministic
        
CONCLUSION: CARTs are not deterministic, so consecutive runs of this program (leaving everything the same), 
             produces different results
"""

import os
import numpy as np
import pandas as pd

import DF_functions
import model_functions
import skML_functions
from class_definitions import *

from sklearn.datasets import load_iris
from sklearn import tree
 
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)



train = np.ones(iris.data.shape[0])
for i in range(len(train)):
    train[i] = i%5

data = np.c_[(iris.data,train)]


for trs in xrange(5):
    print "\n TEST set:",trs,"(i.e., train set excludes set",trs,")"
    train_data = iris.data[data[:,-1] != trs]
    test_data = iris.data[data[:,-1] == trs]
    train_target = iris.target[data[:,-1] != trs]
    test_target = iris.target[data[:,-1] == trs]

    TrainModel_Stats[trs] = {}
    TestModel_Stats[trs] = {}
    
    
    for max_depth in [1,2,3,4]:
        for max_features in [1,2,3,4]:
            kernel = 'md_'+str(max_depth)+"_mf_"+str(max_features)
            predicted_values_train,predicted_values_test = skML_functions.perform_CART(train_data,train_target,test_data,test_target,
                                                                                       DT_max_depth=max_depth, DT_max_features=max_features)    
            train_accuracy = skML_functions.calculate_accuracy(predicted_values_train,train_target)  
            #print "accuracy of prediction on training data",train_accuracy                                                                       
            test_accuracy = skML_functions.calculate_accuracy(predicted_values_test,test_target)
            print "accuracy of prediction on testing data",test_accuracy        

            
            TrainModel_Stats[trs][kernel] = {'accuracy': train_accuracy}  
            TestModel_Stats[trs][kernel]= {'accuracy': test_accuracy}  

models = skML_functions.summarize_model_stats(TestModel_Stats,['accuracy'])
best_avg = 0.0

for model in models:
    print model.name
    avg = 0
    for stat in ['accuracy']:
        print stat,'=', model.__dict__[stat]
        avg += model.__dict__[stat]
    avg = avg/1.0
    if avg/1.0 > best_avg:
        best_model = model.name
        best_avg = avg/1.0
print "best model is:", best_model
for model in models:
    if model.name == best_model:
        for stat in ['accuracy']:
            print stat,'=', model.__dict__[stat]

    
