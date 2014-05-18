# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 17:29:03 2014

@author: ogavril
"""

import os

import pandas as pd
import numpy as np
import statsmodels.api as sm

def write_solution2file(df,MODEL_NAMES,fn=None):
    if fn == None:
        fn = "solution.csv"
    for model_name in MODEL_NAMES:
        fn = model_name+"_"+fn
        fn = open(fn,'w')
        fn.write('customer_ID,plan'+"\n")
        for indx in df.index:
            l = str(indx)+","
            for letter in ['A_2','B_2','C_2','D_2','E_2','F_2','G_2']:
                l += str(int(df['_PREDICTED:'+model_name+":"+letter][indx]))
            fn.write(l+"\n")
        fn.close()
def filling_nas(df,categorical_vars):
    for var in categorical_vars:
        df[var].fillna(-1,inplace=True)
    
    var = 'duration_previous'
    for indx in df.index: 
        df[var][indx] = 0
    
def convert_time(text):
    try:
        hr = int(text.split(':')[0])
        m = float(text.split(':')[1])
        return round(hr+m/60.0,2)
    except:
        print 'could not convert time:',text
        return None

def variables_2_remove(df,vars_name_starts):
    cols2remove = []
    for name in vars_name_starts:
        for col in df.columns:
            if col.startswith(name):
                cols2remove.append(col)
    print "removing columns",cols2remove
    df = df.drop(cols2remove,1)
    return df
                

def summarize_data_intoDict(df):
    """CAUTION: this function assumes that process_df_keeping_last_observations() has been run"""
    
    stationary_vars = ['state','location',"group_size","homeowner","car_age","car_value","risk_factor","age_oldest","age_youngest",
                   "married_couple","C_previous","duration_previous"]    
    ts_vars = ['shopping_pt','record_type','day','time','A','B','C','D','E','F','G','cost']
    recD = {}
    
    
    for i in df.index:
        customer_ID =  df['customer_ID'][i]
    
        if customer_ID not in recD.keys():
            recD[customer_ID] = {}
#            for var in stationary_vars:
#                recD[customer_ID][var] = df[var][i]
            for var in ts_vars:
                recD[customer_ID][var] = [df[var][i]]

        for var in stationary_vars:
#            if recD[customer_ID][var] != df[var][i]: #update with the last
                #print 'discrepancy in',var,recD[customer_ID][var],"<>",df[var][i]
            recD[customer_ID][var] = df[var][i]
        for var in ts_vars:
            recD[customer_ID][var].append(df[var][i])
            
#    for ID in recD.keys():
#        for var in stationary_vars:
#            if str(recD[ID][var]) == 'nan':
#                recD[ID][var] = -1
                    
    for ID in recD.keys():
        for i in range(len(recD[ID]['time'])):
            recD[ID]['time'][i] = convert_time(recD[ID]['time'][i])
    return recD
    

        
def process_df_keeping_last_observations(df,trainingset=1):#if the test set it passed,set trainingset=0
    """if it's a training set, then last 3 observations should be kept--the 3rd one being the response, 
       if it's a test set, then the last 2 observations are to be kept"""
    ID = ""
    indx2remove = []
    currentIndx = []
    for indx in df.index:
        customer_ID =  df['customer_ID'][indx]
        if customer_ID != ID:
            ID = customer_ID
            if currentIndx != []:
                for i in range(len(currentIndx)-2-trainingset):
                    indx2remove.append(currentIndx[i])
            currentIndx = []
        currentIndx.append(indx)
    df = df.drop(indx2remove,0)
    return df      
            
    

def prepare_data_for_analysis(recD,trainingset=True):
    """takes the dictionary where the original data been stored (and cleaned)"""
    stationary_vars = ['state','location',"group_size","homeowner","car_age","car_value","risk_factor","age_oldest","age_youngest",
                       "married_couple","C_previous","duration_previous"]    
    ts_vars = ['shopping_pt','day','time','A','B','C','D','E','F','G','cost']
    
    tmpD = {}
    for ID in recD.keys():
        tmpD[ID] = {}
        for var in recD[ID].keys():
            if var in stationary_vars:
                tmpD[ID][var] = recD[ID][var]
        last = recD[ID]['shopping_pt'][-1]-1 #need to take into account 0-start
        ind1 = last-2 #based on the data we know that there are always at least 2 shopping pts BEFORE a purchase (even for the test data)
        ind2 = last-1
        pts = [ind1,ind2,last]
        if not trainingset:
            pts = [ind2,last]
        for var in ts_vars:
            for i in range(len(pts)):
                tmpD[ID][var+"_"+str(i)] = recD[ID][var][pts[i]]
    return tmpD

def print_dict_2file(Dict,fn=None):
    if fn == None:
        fn = "t_dict.csv"
    fn = open(fn,'w')
    
    headers = None
    for ID in Dict.keys():
        headers = Dict[ID].keys()
        break
    
    if headers != None:
        fn.write("customer_ID,"+",".join(headers)+"\n")
    else:
        print "!!! CAUTIOIN: headers are missing"
    for ID in Dict.keys():
        l = str(ID)+","
        for header in headers:
            l += str(Dict[ID][header]) +","
        fn.write(l+"\n")
    fn.close()
    
def transform_categorical_vars(DF,categorical_vars):
    for var in categorical_vars:
        new_vars = sm.tools.categorical(np.array(DF[var]),drop=True)
        for i in range(new_vars.shape[1]):
            DF[var+':'+str(i)] = pd.Series(new_vars[:,i],index = DF.index)
    return DF
    
def valid_columns(df,target_names,categorical_vars):
    t = []
    for col in df.columns:
        if col not in [ "_TRAIN"]+target_names+categorical_vars:
            if not col.startswith("_"):
                t.append(col)
    return t


def randomize_prediction_v1(df,target_name):
    vals = [elem for elem in df[target_name].unique()]
    probs =[0] +[0 for x in range(len(vals))]
    for v in range(len(vals)):
        count = df[target_name][df[target_name] == vals[v]].count()
        probs[v+1] = probs[v]+ float(count)/float(len(df[target_name]))
    
    print probs    
    random_target = np.array(np.zeros(len(df[target_name])))
    t_ = np.random.uniform(low=0.0,high=1.0,size=len(df[target_name]))
    for i in xrange(len(df[target_name])):
        for v in xrange(1,len(probs)):
            if probs[v-1] <= t_[i] and t_[i] <= probs[v]:
                random_target[i] = vals[v-1]
                break
    return random_target

        


#for var in ['shopping_pt']:    
#    plot_functions.hist_plot(df[var],title=var,figname=var+'_playing')        
                    
                
