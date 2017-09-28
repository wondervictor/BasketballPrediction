# -*- coding: utf-8 -*-

"""
Evalute the model
"""
from sklearn import metrics

def auc(y, pred_y):
    return metrics.roc_auc_score(y, pred_y)
    
