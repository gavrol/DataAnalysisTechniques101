# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 08:09:51 2014

@author: olena
PURPOSE: stuff for finding eigen vectors

"""

import numpy as np
import scipy as sp
import random
from sklearn import decomposition
import plot_functions
import statsmodels.api as sm


def create_response_var(X):
    beta = [0.5,-0.01,5.0,-1.0,1.0]
    sig = 0.5
    y_True = np.dot(X,beta)
    e = np.random.normal(size=X.shape[0])
    w = np.ones(X.shape[0])
    y = y_True +sig*w*e
    return y,y_True   

# create a few vectors
sample_size = 20
x1 = np.linspace(0,10,sample_size) #get me 20 numbers that are evenly spread from 0 to 10
x2 = x1 + np.random.normal(loc=0,scale=1,size=len(x1)) 
#could have done it this way x2 = x1 + [random.normal() for i in range(len(x1))   ]
x3 = np.random.exponential(scale=2.0,size=len(x1))
x4 = np.random.uniform(low=-0.5,high=1.3,size=len(x1))
x5 = x4 + x3 + np.random.uniform(low=0,high=1,size=len(x1))

#now put them in a matrix by concatinating them as columns

M = np.c_[(x1,x2,x3,x4,x5)]
#here is an alternative way of doing it

mX = np.c_[(x1)]
features = [x1,x2,x3,x4,x5]
for c in range(1,len(features)):
    mX = np.c_[(mX,features[c])]
    
print mX
print "mX.shape =", mX.shape

c = 1
while c <= len(features):
    pca = decomposition.PCA(n_components=c)
    pca.fit_transform(mX)
    print "explained variance ratios with",c,"components:",pca.explained_variance_ratio_
    if sum(pca.explained_variance_ratio_) > 0.96:
        break
    c += 1
num_components = min(c,len(features))
print "finished PCA, the best number of components is",num_components
#    pcaN = decomposition.PCA()


"""now let's do some eigen decomposition"""
covMat = np.cov(mX.T) #why mX.T? because we are doing decomposition in terms of features, of which there are as many as len(features)
print "check the size of the cov matrix",covMat.shape
EigenStuff = np.linalg.eigh(covMat) 
eigen_vals = EigenStuff[0] #values, which are usually sorted
eigen_vecs = EigenStuff[1] #vectors are here
print "the eigen vectors are:",eigen_vecs

print "The eigen vectors are actually the columns of the above matrix, NOT rows"

"""even though the eigen value and vectors should be sorted accordingly, 
I don't trust it so I do the following:"""
eigen_tups = zip(eigen_vals,range(len(eigen_vals)))
eigen_tups.sort(reverse=True) #now sort

PCA_eigen_vecs = np.array([])
for c in range(num_components):
    eigen_vec = np.array([eigen_vecs[i][eigen_tups[c][1]] for i in range(len(eigen_vals))])
    eigen_vec = eigen_vec.reshape(len(eigen_vals),1)
    if c == 0:
        PCA_eigen_vecs = np.c_[(eigen_vec)]
    else:
        PCA_eigen_vecs = np.c_[(PCA_eigen_vecs,eigen_vec)]
    print "the eigen vector under consideration is the",eigen_tups[c][1]+1,"-th vector amongst",len(eigen_vals)

print "the chosen eigen vector(s) are",PCA_eigen_vecs
print "sorted eigen values are:", eigen_tups


newX = np.dot(mX,PCA_eigen_vecs)
print newX

y,y_True = create_response_var(mX)

X1 = np.c_[(mX,np.ones(mX.shape[0]))]

mod = sm.OLS(y,mX).fit()
print "Fitted OLS\n",mod.summary()    
plot_functions.plots(mod,y,x1,len(x1),y_true=y_True,figfn ="OLS")

newX  = np.c_[(newX,np.ones(mX.shape[0]))]
mod_pca = sm.OLS(y,newX).fit()
print "Fitted OLS\n",mod_pca.summary()    
plot_functions.plots(mod_pca,y,x1,len(x1),y_true=y_True,figfn ="OLS_afterPCA_transformation")
