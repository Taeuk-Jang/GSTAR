import numpy as np

def complete_dict(per_dict):
    per_dict['FP'] = per_dict['FP_priv'] + per_dict['FP_unpriv']
    per_dict['TP'] = per_dict['TP_priv'] + per_dict['TP_unpriv'] 
    per_dict['FN'] = per_dict['FN_priv'] + per_dict['FN_unpriv'] 
    per_dict['TN'] = per_dict['TN_priv'] + per_dict['TN_unpriv'] 
    

def eq_opp(per_dict):
    return abs(tpr(per_dict, 1) - tpr(per_dict,0))

def fpr_diff(per_dict):
    return abs(fpr(per_dict, 1) - fpr(per_dict,0))

def eq_odds(per_dict):
    return (eq_opp(per_dict) + fpr_diff(per_dict))/2

def tpr(per_dict, priv = 2):
    if priv == 1:
        return per_dict['TP_priv']/(per_dict['TP_priv'] + per_dict['FN_priv'])
    elif priv == 0:
        return per_dict['TP_unpriv']/(per_dict['TP_unpriv'] + per_dict['FN_unpriv'])
    elif priv == 2:
        return per_dict['TP']/(per_dict['TP'] + per_dict['FN'])
    
def fpr(per_dict, priv = 2):
    if priv == 1:
        return per_dict['FP_priv']/(per_dict['FP_priv'] + per_dict['TN_priv'])
    elif priv == 0:
        return per_dict['FP_unpriv']/(per_dict['FP_unpriv'] + per_dict['TN_unpriv'])
    elif priv == 2:
        return per_dict['FP']/(per_dict['FP'] + per_dict['TN'])
    
def acc(per_dict, priv = 2):
    if priv == 1:
        return (per_dict['TN_priv'] + per_dict['TP_priv'])/(per_dict['TP_priv'] + per_dict['FP_priv'] + per_dict['TN_priv'] + per_dict['FN_priv'])
    elif priv == 0:
        return (per_dict['TN_unpriv'] + per_dict['TP_unpriv'])/(per_dict['TP_unpriv'] + per_dict['FP_unpriv'] + per_dict['TN_unpriv'] + per_dict['FN_unpriv'])
    elif priv == 2:
        return (per_dict['TN'] + per_dict['TP'])/(per_dict['TP'] + per_dict['FP'] + per_dict['TN'] + per_dict['FN'])
    
def bal_acc(per_dict, priv = 2):
    return (tpr(per_dict, priv) + (1-fpr(per_dict,priv)))/2

def confusion_tpr(z, priv=2):
    if priv == 1:
        return (z[0]/sum(z[[0,1]])).item()
    elif priv == 0:
        return (z[4]/sum(z[[4,5]])).item()
    elif priv == 2:
        return (sum(z[[0,4]])/sum(z[[0,1, 4,5]])).item()
    
def confusion_tpr_diff(z):
    return abs(confusion_tpr(z, 1) - confusion_tpr(z, 0))

def confusion_fpr_diff(z):
    return abs(confusion_fpr(z, 1) - confusion_fpr(z, 0))
    
def confusion_fpr(z, priv=2):
    if priv == 1:
        return (z[2]/sum(z[[2,3]])).item()
    elif priv == 0:
        return (z[6]/sum(z[[6,7]])).item()
    elif priv == 2:
        return (sum(z[[2,6]])/sum(z[[2,3, 6,7]])).item()
    
def confusion_eqodd(z):
    return confusion_tpr_diff(z) + confusion_fpr_diff(z)
    
def confusion_acc(z, priv=2):
    if priv == 1:
        return (sum(z[[0,3]])/sum(z[0:4])).item()
    elif priv == 0:
        return (sum(z[[4,7]])/sum(z[4:8])).item()
    elif priv == 2:
        return (sum(z[[0,3, 4,7]])/sum(z)).item()
    
def confusion_bal_acc(z, priv=2):
    return (confusion_tpr(z, priv) + (1-confusion_fpr(z, priv)))/2

def confusion_bal_acc_diff(z):
    return abs(confusion_bal_acc(z, 1) - confusion_bal_acc(z, 0))
    
def confusion_predictive(z, priv):
    #Y_hat = 1 | A = priv
    if priv == 1:
        return sum(z[[0, 2]])/sum(z[[0,1,2,3]])
    elif priv == 0:
        return sum(z[[4, 6]])/sum(z[[4,5,6,7]])
    elif priv == 2:
        return sum(z[[0,2,4,6]])
def confusion_dimp(z):
    return confusion_predictive(z, 0)/confusion_predictive(z, 1)
    
def confusion_stat_diff(z):
    return abs(confusion_predictive(z, 0) - confusion_predictive(z, 1))
    
    
    
    