# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:13:40 2020

@author: guseh
"""
import numpy as np
import os
try:
    import cPickle as pickle
except BaseException:
    import pickle
    
def rmsse(y_true, y_pred, y_hist, axis = None, weight = None):
    '''
    y_true: 실제 값 
    y_pred: 예측 값
    y_hist: 과거 값 (public LB는 v1 기간으로 계산, private LB는 v2 기간으로 계산)
    '''
    
    y_true, y_pred, y_hist = np.array(y_true), np.array(y_pred), np.array(y_hist)
    
    h, n = len(y_true), len(y_hist)


    numerator = np.sum((y_true - y_pred)**2, axis = axis)
    
    denominator = 1/(n-1)*np.sum((y_hist[1:] - y_hist[:-1])**2, axis = axis)
    
    msse = 1/h * numerator/denominator
    
    rmsse = msse ** 0.5
    
    score = rmsse
    
    if weight is not None:
        
        score = rmsse.dot(weight)
    
    return score

def save_obj(obj, name):
    try:
        with open('results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        os.mkdir('results')
        with open('results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)        

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials
    

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param

def trans(dataset, start_index, end_index, past, future, x_columns, y_columns):
    dataset.index = range(dataset.shape[0])
    data = []
    labels = []
    
    start_index = start_index + past
    
    if end_index is None:
        end_index = dataset.shape[0]
    
    for i in range(start_index, end_index-future):
        indices = np.array(dataset.loc[i-past:i, x_columns])
        data.append(indices)
        
        labels.append(np.array(dataset.loc[i+future, y_columns]))
        
    data = np.array(data)
    data = data.reshape(data.shape[0], -1)
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0],-1)
    
    return data, labels