# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 21:45:08 2014

@author: ogavril
"""
import numpy as np
import statsmodels.api as sm

import plot_functions
import common_functions

from class_definitions import * #needed for INDEP_VARIABLE

def OLS_iteration(df,yVarN,xVarN,pltname=None):
    lin_model = sm.OLS(df[yVarN],df[xVarN]).fit()
    print "\nOLS using",len(xVarN),"variables"
    print lin_model.summary()
    print "model's MSE:",lin_model.mse_resid
    residuals = lin_model.resid
    y_pred = lin_model.predict()
    if pltname != None:
        plot_functions.plots2_TrueVSpred_Resid(df[yVarN],y_pred,residuals=residuals,weights=None,y_true=None,figfn =pltname)
    return lin_model
    
def GLM_iteration(df,yVarN,xVarN,pltname=None,family_= sm.families.Gaussian()):
    glm_model = sm.GLM(df[yVarN],df[xVarN],family=family_ ).fit()
    print "\nGLM using",len(xVarN),"variables"
    print glm_model.summary()
    residuals = glm_model.resid_response
    y_pred = glm_model.predict()
    if pltname != None:
        plot_functions.plots2_TrueVSpred_Resid(df[yVarN],y_pred,residuals=residuals,weights=None,y_true=None,figfn =pltname)
    return glm_model

def Logit_iteration(df,yVarN,xVarN,pltname=None):
    log_model = sm.Logit(df[yVarN],df[xVarN]).fit()
    print "\nLogit using",len(xVarN),"variables"
    print log_model.summary()
    y_pred = log_model.predict()
    if pltname != None:
        plot_functions.plot_true_vs_pred_logit(df[yVarN],y_pred,title="True vs. Predicted LogReg",figname=pltname)
    return log_model
    
def evaluate_SMmodel_on_testSet(mod,df,yVarN):
    print "evaluting the model based on a test set"
    coeff = {}
    for xN in mod.params.keys():
        coeff[xN] = mod.params[xN]

    y_pred = np.array([0 for i in xrange(len(df.index))])
    for col in df.columns:
        if col in coeff.keys():
            t_ = np.array(df[col]).reshape(y_pred.shape[0])
            y_pred = coeff[col]*t_ + y_pred
    y_true = np.array(df[yVarN]).reshape(y_pred.shape[0])
    MSE  = common_functions.MSE(y_pred,y_true)
    return MSE

def calculate_yPred_logit(log_model,df,yVarN):
    """evaluating based on coefficients, make sure 'const' is included in df, else you need to add it manually"""
    
    coeff = {}
    for xN in log_model.params.keys():
        coeff[xN] = log_model.params[xN]

    y_pred_lin = np.array([0 for i in xrange(len(df.index))])
    for col in df.columns:
        if col in coeff.keys():
            t_ = np.array(df[col]).reshape(y_pred_lin.shape[0])
            y_pred_lin = coeff[col]*t_ + y_pred_lin
    y_pred = 1.0/(1.0+np.exp(-y_pred_lin)) #this step is the sigmoid transformation
    return y_pred

def get_model_coefficients(model):
    coeff_D = {}
    xVars = [key for key in model.params.keys()]
    for xN in xVars:
        coeff_D[xN] = {'mean': model.params[xN]}
    for n in xrange(len(xVars)):
        coeff_D[xVars[n]]['CI_lower'] = model.conf_int()[0][n]
        coeff_D[xVars[n]]['CI_upper'] = model.conf_int()[1][n]
        
    return coeff_D
    

def calculate_yPred_basedOn_1xVar_logit(log_model,df,xVarN,yVarN):
    """used to estimate a predictive power of just one explanatory variable """
    coeff = {}
    for xN in log_model.params.keys():
        coeff[xN] = log_model.params[xN]
    y_pred_lin = np.array([0 for i in xrange(len(df.index))])
    
    if xVarN not in coeff.keys():
        print "!!! ERROR: variable",xVarN,"is NOT in the model"
        return None
    else:
        for col in [xVarN,'const']:
            if col in coeff.keys():
                t_ = np.array(df[col]).reshape(y_pred_lin.shape[0])
                y_pred_lin = coeff[col]*t_ + y_pred_lin
        y_pred = 1.0/(1.0+np.exp(-y_pred_lin)) #this step is the sigmoid transformation
        return y_pred
    
def calculate_SensSpecifPrec(df_,log_model,yVarN,TH=0.5,is_test_set=False):
    """given a threashold TH, evaluate sensitivity, specificity and precision
    Sensitivity and specificity are statistical measures of the performance of a binary classification test, 
    also known in statistics as classification function. 
    Sensitivity (also called the true positive rate, or the recall rate in some fields) 
    measures the proportion of actual positives which are correctly identified as such 
    (e.g. the percentage of sick people who are correctly identified as having the condition). 
    Specificity (sometimes called the true negative rate) measures the proportion of negatives which are correctly 
    identified as such (e.g. the percentage of healthy people who are correctly identified as not having the condition).  
    A perfect predictor would be described as 100% sensitive (i.e. predicting all people from the sick group as sick) and 100% specific (i.e. not predicting anyone from the healthy group as sick);    
    """
    df = df_.copy()
    df['pred'] = calculate_yPred_logit(log_model,df,yVarN)
    if not is_test_set: #only a senity check
        df['model_pred'] = log_model.predict() #doesn't exist for test set
        comp_array = np.array(df['pred'] - df['model_pred'])
        if abs(comp_array.min() -0) > 1e-5 or abs(comp_array.max() -0) > 1e-5:
            print "\n\n!!!ERROR: check prediction in calculate_SensSpecifPrec()\n"
            print comp_array.min(),"~?",comp_array.max()
    return SensSpecifPrec(df,yVarN,TH=TH)
        
def SensSpecifPrec(df,yVarN,TH=0.5):
    """IMPORTANT: the dataframe MUST have columns named 'pred' """
    if 'pred' in df.columns:
        numTP = df['pred'][(df['pred']>TH) &(df[yVarN]==1)].count()
        numFP = df['pred'][(df['pred']>TH) &(df[yVarN]==0)].count()
        numTN = df['pred'][(df['pred']<=TH) &(df[yVarN]==0)].count() 
        numFN = df['pred'][(df['pred']<=TH) &(df[yVarN]==1)].count()
        #also numTP+numFN = df[yVarN][df[yVarN]==1].sum()
        sensitivity = float(numTP)/float(max(numTP+numFN,1))
        specificity = float(numTN)/float(max(numTN+numFP,1))
        precision = float(numTP)/float(max(numTP+numFP,1))
        accuracy = float(numTP +numTN)/float(numTP +numTN +numFP +numFN)
        return sensitivity,specificity,precision,accuracy
    else:
        return None

def calculate_SensSpecifPrecAccur(target_predicted,target):
    
    if len(target_predicted) == len(target):
        numTP = len(target_predicted[(target_predicted==1) & (target==1)])
        numFP = len(target_predicted[(target_predicted==1) & (target==0)])
        numTN = len(target_predicted[(target_predicted==0) & (target==0)])
        numFN = len(target_predicted[(target_predicted==0) & (target==1)])
        sensitivity = float(numTP)/float(max(numTP+numFN,1))
        specificity = float(numTN)/float(max(numTN+numFP,1))
        precision = float(numTP)/float(max(numTP+numFP,1))
        accuracy = float(numTP +numTN)/float(numTP +numTN +numFP +numFN)
        return sensitivity,specificity,precision,accuracy
    else:
        return None
        
    
def calculate_randomized_prediction(df,rate):
    df['pred'] = 0
    for i in xrange(len(df.index)):
        if i% int(1./rate) == 0:
            df['pred'][i]  = 1
    return df

def randomize_prediction_v1(df,success_rate):
    df['pred'] = 0
    t_ = np.random.uniform(low=0.0,high=1.0,size=len(df.index))
    for i in xrange(len(df.index)):
        if t_[i] <= success_rate:
            df['pred'][i] = 1
    return df
    
def evaluate_model_effectiveness_logit(df,yVar,log_model,TH=0.5,is_test_set=False):
    """this function evaluates effectiveness of a logistic model;
        Assumptions: a) yVar is 0/1;
                     b) logistic regression has been run and the results of the model are in log_model object
    """
    tset = "Train"
    if is_test_set == True: tset = 'Test'
    print "\n evaluating model's effectiviness on",tset,"set"
    numYES = df[yVar][df[yVar]==1].count()
    numObs = len(df.index)
    probYES = float(numYES)/float(numObs)
    #print numYES,numObs,probYES

    sensitivity,specificity,precision,accuracy = calculate_SensSpecifPrec(df,log_model,yVar,TH=TH,is_test_set=is_test_set)
    print "For",tset,"set with TH=",TH,"sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)

    tdf = df.copy()
    #tdf = calculate_randomized_prediction(tdf,probYES)
    tdf = randomize_prediction_v1(tdf,probYES)
    
    AdHsensitivity,AdHspecificity,AdHprecision,AdHaccuracy = SensSpecifPrec(tdf,yVar,TH=TH) 
    print "For AdHoc prediction sensivity %f\n specificity %f\n precision %f\n accuracy %f \n" %(AdHsensitivity,AdHspecificity,AdHprecision,AdHaccuracy)
    print "model improvement",round(precision/AdHprecision,2)

def evaluate_model_coeff_stability(variables):
    for var in variables:
        if var.gap != None:
            print "!!! CAUTION:",var,"has inconsistent coefficient estimates. The gap is (",var.gap[0],",",var.gap[1],",)"
    
            
def gather_model_coeff(TrainSet_Model_Coeff):
    model_xVars = {}    
    variables = {}
    train_set = TrainSet_Model_Coeff.keys()
    for trs in train_set:
        for var in TrainSet_Model_Coeff[trs].keys():
            if var not in model_xVars.keys():
                model_xVars[var] = {}
        
    for var in model_xVars.keys():
        variable = INDEP_VARIABLE(var)
        for t in xrange(len(train_set)):           
            model_xVars[var][train_set[t]] = {}
            if var in TrainSet_Model_Coeff[train_set[t]].keys():
                model_xVars[var][train_set[t]]['mean'] = TrainSet_Model_Coeff[train_set[t]][var]['mean']
                model_xVars[var][train_set[t]]['segStart'] = TrainSet_Model_Coeff[train_set[t]][var]['CI_lower']
                model_xVars[var][train_set[t]]['segEnd'] = TrainSet_Model_Coeff[train_set[t]][var]['CI_upper']
            else:
                model_xVars[var][train_set[t]]['mean'] = 0
                model_xVars[var][train_set[t]]['segStart'] = 0
                model_xVars[var][train_set[t]]['segEnd'] = 0
        starts = [model_xVars[var][trs]['segStart'] for trs in model_xVars[var].keys() ] 
        ends = [model_xVars[var][trs]['segEnd'] for trs in model_xVars[var].keys() ] 
        variable.CI_starts = [elem for elem in starts]
        variable.CI_ends = [elem for elem in ends]
        variable.means = [model_xVars[var][trs]['mean'] for trs in model_xVars[var].keys()   ]
        starts.sort()
        ends.sort()
        if starts[-1] > ends[0]:
            variable.gap = (ends[0],starts[-1])
        else:
            variable.estimated_range = (starts[-1],ends[0])
        variables[var] = variable
    return variables
        
    
def cross_validation_logit(df,yVar,xVarN,trainVarN='train',TH=0.5,dataSetName="TestX"):
    print "\n\n CROSS VALIDATION stage..." 

    TrainSet_Model_Coeff = {}
    yVarN = [yVar]
    train_set = df[trainVarN].unique()
    for trs in train_set:

        print "\n TEST set:",trs,"(i.e., train set excludes set",trs,")"
        trDF = df[df['train']!=trs]
        testDF = df[df['train']==trs]
        log_model = Logit_iteration(trDF,yVarN,xVarN,pltname=None) #dataSetName+"_CVtrainSet_"+str(trs))
        sensitivity,specificity,precision,accuracy = calculate_SensSpecifPrec(trDF,log_model,yVar,TH=TH,is_test_set=False)
        #print "For Training sets (excl.",trs,") sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)
        evaluate_model_effectiveness_logit(trDF,yVar,log_model,TH=TH,is_test_set=False)

        TrainSet_Model_Coeff[trs] = get_model_coefficients(log_model) 
        
        sensitivity,specificity,precision,accuracy = calculate_SensSpecifPrec(testDF,log_model,yVar,TH=TH,is_test_set=True)
        #print "For Test set",trs,"sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)
        evaluate_model_effectiveness_logit(testDF,yVar,log_model,TH=TH,is_test_set=True)
    variables = gather_model_coeff(TrainSet_Model_Coeff)
    return variables
