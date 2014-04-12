# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 08:09:51 2014

@author: olena
PURPOSE: stuff for finding eigen vectors

"""

import numpy as np
import scipy as sp
import pandas as pd
import datetime


def create_response_var_4_test01(X):
    print 'in creating responce function with X of shape',X.shape    
    beta = [0.5,-0.01,5.0,-1.0,1.0]
    sig = 0.5
    y_True = np.dot(X,beta)
    e = np.random.normal(size=X.shape[0])
    w = np.ones(X.shape[0])
    y = y_True +sig*w*e
    return y,y_True   

def create_response_var_4_test02(X):
    print 'in creating responce function with X of shape',X.shape
    beta = [0.05,-0.05,0.5,-3.0,-10.5]
    sig = 0.9
    y_True = np.dot(X,beta)
    e = np.random.normal(loc=0.0,scale=3,size=X.shape[0])
    #e = np.random.triangular(-2,0,2,size=X.shape[0])
    w = np.ones(X.shape[0])
    y = y_True +sig*w*e
    return y,y_True   

def create_df(y_var,features,fn = None ):
    ser = {}
    for c in xrange(len(features)):
        s = 'f'+str(c+1)
        ser[s] = features[c]
    
    ser['response'] = y_var
    
    df = pd.DataFrame(ser)
    if fn == None:
        fn = "Default_"+datetime.datetime.now().date().strftime("%Y-%m-%d")+".csv"
    df.to_csv(fn)
        


# create a few vectors
sample_size = 400
x1 = np.linspace(0,10,sample_size) #get me 20 numbers that are evenly spread from 0 to 10
x2 = x1 + np.random.normal(loc=0,scale=1,size=len(x1)) 
#could have done it this way x2 = x1 + [random.normal() for i in range(len(x1))   ]
x3 = np.random.exponential(scale=2.0,size=len(x1))
x4 = np.random.uniform(low=-0.5,high=1.3,size=len(x1))
x5 = x4 + x3 + np.random.uniform(low=0,high=1,size=len(x1))
x2x3 = np.multiply(x2,x3)
x7 = np.random.exponential(scale=10.0,size=sample_size)
x7lg = np.log1p(x7)
x2sq = np.multiply(x2,x2)


#now put them in a matrix by concatinating them as columns
#M = np.c_[(x1,x2,x3,x4,x5)]
#here is an alternative way of doing it
"""creating text01"""
mX = np.c_[(x1)]
features = [x1,x2,x3,x4,x5]
for c in range(1,len(features)):
    mX = np.c_[(mX,features[c])]
print "mX.shape =", mX.shape

y,y_True = create_response_var_4_test01(mX)
create_df(y,features,fn ="../data/test01.csv" )

"""creating test04"""
Xa = np.c_[(x2)]
features = [x2,x2sq,x2x3,x5,x7lg]
for c in range(1,len(features)):
    Xa = np.c_[(Xa,features[c])]   
y,y_True = create_response_var_4_test02(Xa)

features = [x1,x2,x3,x4,x5,x7]
create_df(y,features,fn ="../data/test04.csv" )

