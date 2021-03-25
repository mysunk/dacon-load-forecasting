# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:44:41 2020

@author: guseh
"""
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from util import *
from functools import partial
from sklearn.ensemble import RandomForestRegressor

#%% load dataset
# target = load_dataset_v1(features = ['datetime','area','temp'])
submission = pd.read_csv('data_raw/AIFrienz S3_v2/sample_submission.csv')
target_w = pd.read_csv('data/target_v2.csv')
ref = pd.read_csv('submit/sub_baseline_v3_sw_fin.csv')
ref['date'] = pd.to_datetime(ref['date'])
target = pd.read_csv('data/target_oil.csv')

target['year'] = target_w['year']
target['month'] = target_w['month']
target['day'] = target_w['day']
target['dayofweek'] = target_w['dayofweek']
target['temp_min'] = target_w['temp_min']
target['temp_max'] = target_w['temp_max']
target['temp_mean'] = target_w['temp_mean']
target['date'] = pd.to_datetime(target['date'])

#%% load testset
submission['date'] = pd.to_datetime(submission['date'])
submission_top_half = submission.loc[:27,:].copy()
submission = submission.loc[28:, :]
test = submission.copy()
test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['dayofweek'] = test['date'].dt.dayofweek


#%% SUPPLY 예측

x_columns = ['supply','year','month','day','dayofweek','temp_min','temp_max','temp_mean']
past = 28 - 1

pasts = [past,0,0,0,0,past,past,past]
y_column = ['supply']

# 예측할 데이터 변형
test_x = []
for j,x_column in enumerate(x_columns):
    data = []
    dataset_sub = target.loc[target.shape[0]-pasts[j]-1:,x_column].values.copy()
    for element in dataset_sub:
        test_x.append(element)
test_x = np.ravel(test_x)
test_x = test_x.reshape(1,-1)

results = []
for random_state in [0,2,4,5,7,9]:
    print(f'===={random_state}=====')
    prediction = []
    prediction_losses = []
    prediction_losses_std = []
    for future in range(7,35):
        trials = load_obj('0521_3/'+y_column[0])
        params = trials[19]['params']
        params['random_state'] = random_state
        nfolds = 10
        
        preds = []
        train, train_label = trans(target, 0, None, pasts, future, x_columns, y_column)
        losses = np.zeros((nfolds,2))
        kf = KFold(n_splits=nfolds,random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
            x_train = train[train_index]
            y_train = train_label[train_index]
            x_test = train[test_index]
            y_test = train_label[test_index]
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_test, label=y_test)
            rmmse_obj = partial(rmsse_lgb,y_hist = target[y_column[0]])
            model = lgb.train(params, train_set = dtrain,  
                              valid_sets=[dtrain, dvalid],num_boost_round=1000,verbose_eval=False,feval=rmmse_obj,
                                     early_stopping_rounds=5)
            losses[i,0] = model.best_score['training']['rmsse_modified']
            losses[i,1] = model.best_score['valid_1']['rmsse_modified']
            preds.append(model.predict(test_x))
        y_pred = np.mean(preds,axis=0)
        prediction.append(y_pred)
        prediction_losses.append(np.mean(losses,axis=0))
        prediction_losses_std.append(np.std(losses,axis=0))
        print(f'Mean is {np.mean(losses,axis=0)} and std is {np.std(losses,axis=0)}')
    print(np.mean(prediction_losses,axis=0))
    print(np.mean(prediction_losses_std,axis=0))
    
    result = np.concatenate(prediction,axis=0)
    results.append(result)

test['supply'] = np.mean(results,axis=0)

#%%
plt.plot(target.loc[681:,'date'],target.loc[681:,'supply'])
plt.plot(test.loc[:,'date'],test.loc[:,'supply'])
plt.plot(ref.loc[28:,'date'],ref.loc[28:,'supply'])

#%% SMP 예측 -- mean
x_columns  = ['supply','year','month','day','dayofweek','temp_min','temp_max','temp_mean','smp_min','smp_max','smp_mean',
              'lng','gold','wti','du','brent','usd']   

past = 2 - 1
pasts = [past ,0,0,0,0,past ,past ,past ,past ,past ,past,
         past,past,past,past,past,past,past,past,past]

# 예측할 데이터 변형
test_x = []
for j,x_column in enumerate(x_columns):
    data = []
    dataset_sub = target.loc[target.shape[0]-pasts[j]-1:,x_column].values.copy()
    for element in dataset_sub:
        test_x.append(element)

test_x = np.ravel(test_x)
test_x = test_x.reshape(1,-1)

# pasts = [28,0,0,0,0]
y_column = ['smp_mean']

# default
params = {
    'metric': 'None',
    'njobs':-1,
    'learning_rate': 0.1,
    #   'bagging_freq':23,
    #   'boosting':'gbdt',
    #   'colsample_bynode':0.8,
    #   'colsample_bytree':0.7,
    #   'max_bin':9,
    # 'reg_alpha':0.1,
    #   'reg_lambda':0.1,
    #     'min_sum_hessian_in_leaf':0.05,
    #     'num_leaves':33,
    #   'max_depth':-1,
    }

# 학습
results = []
for random_state in [0]:
    prediction = []
    prediction_losses = []
    params['seed'] = random_state
    for future in range(7,35):
        nfolds = 5
        preds = []
        train, train_label = trans(target, 0, None, pasts, future, x_columns, y_column)
        train = train[-157:,:]
        train_label = train_label[-157:] # 12월 14일 것 부터만..
        
        losses = np.zeros((nfolds,2))
        kf = KFold(n_splits=nfolds,random_state=None, shuffle=False)
        # plt.figure()
        for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
            x_train = train[train_index]
            y_train = train_label[train_index]
            x_test = train[test_index]
            y_test = train_label[test_index]
            
            # lgb
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_test, label=y_test)
            rmmse_obj = partial(rmsse_lgb,y_hist = target[y_column[0]])
            model = lgb.train(params, train_set = dtrain,  
                              valid_sets=[dtrain, dvalid],num_boost_round=1000,verbose_eval=False,feval=rmmse_obj,
                                      early_stopping_rounds=5)
            losses[i,0] = model.best_score['training']['rmsse_modified']
            losses[i,1] = model.best_score['valid_1']['rmsse_modified']
            
            preds.append(model.predict(test_x))
        
        y_pred = np.mean(preds,axis=0)
        prediction.append(y_pred)
        prediction_losses.append(np.mean(losses,axis=0))
        print(f'Mean is {np.mean(losses,axis=0)} and std is {np.std(losses,axis=0)}')
    print(np.mean(prediction_losses,axis=0))
    result = np.concatenate(prediction,axis=0)
    results.append(result)
    
test['smp_mean'] = np.mean(results,axis=0)

#%%
plt.plot(target.loc[:,'date'],target.loc[:,'smp_mean'])
plt.plot(test.loc[:,'date'],test.loc[:,'smp_mean'])
# plt.plot(ref.loc[28:,'date'],ref.loc[28:,'smp_mean'])

#%% SMP 예측 -- max
x_columns = ['supply','year','month','day','dayofweek','temp_min','temp_max','temp_mean','smp_min','smp_max','smp_mean']

past = 28 - 1
pasts = [past ,0,0,0,0,past ,past ,past ,past ,past ,past]

# 예측할 데이터 변형
test_x = []
for j,x_column in enumerate(x_columns):
    data = []
    dataset_sub = target.loc[target.shape[0]-pasts[j]-1:,x_column].values.copy()
    for element in dataset_sub:
        test_x.append(element)
test_x = np.ravel(test_x)
test_x = test_x.reshape(1,-1)

y_column = ['smp_max']

# 학습
results = []
for random_state in range(10):
    prediction = []
    prediction_losses = []
    for future in range(7,35):
        trials = load_obj('0521_3/smp_max')
        params = trials[0]['params']
        nfolds = 10
        params['random_state'] = random_state
        preds = []
        train, train_label = trans(target, 0, None, pasts, future, x_columns, y_column)
        # train = train[681:,:]
        # train_label = train_label[681:]
        
        losses = np.zeros((nfolds,2))
        kf = KFold(n_splits=nfolds,random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
            x_train = train[train_index]
            y_train = train_label[train_index]
            x_test = train[test_index]
            y_test = train_label[test_index]
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_test, label=y_test)
            rmmse_obj = partial(rmsse_lgb,y_hist = target[y_column[0]])
            model = lgb.train(params, train_set = dtrain,  
                              valid_sets=[dtrain, dvalid],num_boost_round=1000,verbose_eval=False,feval=rmmse_obj,
                                     early_stopping_rounds=10)
            losses[i,0] = model.best_score['training']['rmsse_modified']
            losses[i,1] = model.best_score['valid_1']['rmsse_modified']
            preds.append(model.predict(test_x))
        y_pred = np.mean(preds,axis=0)
        prediction.append(y_pred)
        prediction_losses.append(np.mean(losses,axis=0))
        print(f'Mean is {np.mean(losses,axis=0)} and std is {np.std(losses,axis=0)}')
    print(np.mean(prediction_losses,axis=0))
    result = np.concatenate(prediction,axis=0)
    results.append(result)

test['smp_max'] = np.mean(results,axis=0)

#%% result
plt.plot(target.loc[681:,'date'],target.loc[681:,'smp_max'])
plt.plot(test.loc[:,'date'],test.loc[:,'smp_max'])
plt.plot(ref.loc[28:,'date'],ref.loc[28:,'smp_max'])

#%% SMP 예측 -- min
x_columns  = ['supply','year','month','day','dayofweek','temp_min','temp_max','temp_mean','smp_min','smp_max','smp_mean',
                  'lng','gold','wti','du','brent','usd']

past = 3 - 1
pasts = [past ,0,0,0,0,past ,past ,past ,past ,past ,past,
         past,past,past,past,past,past,past,past,past]


# 예측할 데이터 변형
test_x = []
for j,x_column in enumerate(x_columns):
    data = []
    dataset_sub = target.loc[target.shape[0]-pasts[j]-1:,x_column].values.copy()
    for element in dataset_sub:
        test_x.append(element)
test_x = np.ravel(test_x)
test_x = test_x.reshape(1,-1)

# default
params = {
    'metric': 'None',
    'njobs':-1,
    'learning_rate': 0.1,
    }

y_column = ['smp_min']

# 학습
prediction = []
prediction_losses = []
for future in range(7,35):
    nfolds = 5
    preds = []
    train, train_label = trans(target, 0, None, pasts, future, x_columns, y_column)
    train = train[-157:,:]
    train_label = train_label[-157:]
    losses = np.zeros((nfolds,2))
    kf = KFold(n_splits=nfolds,random_state=None, shuffle=False)
    for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
        x_train = train[train_index]
        y_train = train_label[train_index]
        x_test = train[test_index]
        y_test = train_label[test_index]
        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(x_test, label=y_test)
        rmmse_obj = partial(rmsse_lgb,y_hist = target[y_column[0]])
        model = lgb.train(params, train_set = dtrain,  
                          valid_sets=[dtrain, dvalid],num_boost_round=1000,verbose_eval=False,feval=rmmse_obj,
                                 early_stopping_rounds=5)
        losses[i,0] = model.best_score['training']['rmsse_modified']
        losses[i,1] = model.best_score['valid_1']['rmsse_modified']
        preds.append(model.predict(test_x))
    y_pred = np.mean(preds,axis=0)
    prediction.append(y_pred)
    prediction_losses.append(np.mean(losses,axis=0))
    print(f'Mean is {np.mean(losses,axis=0)} and std is {np.std(losses,axis=0)}')
print(np.mean(prediction_losses,axis=0))

# Compare results
result = np.concatenate(prediction,axis=0)
test['smp_min'] = result

#%% result
plt.plot(target.loc[681:,'date'],target.loc[681:,'smp_min'])
plt.plot(test.loc[:,'date'],test.loc[:,'smp_min'])
# ref['date'] = pd.to_datetime(ref['date'])
plt.plot(ref.loc[28:,'date'],ref.loc[28:,'smp_min'])
plt.plot(ref.loc[28:,'date'],(ref.loc[28:,'smp_min']+test.loc[:,'smp_min'])/2)


#%% plot result
for col in ['smp_max','smp_min','smp_mean','supply']:
    plt.figure()
    plt.plot(ref.loc[28:,col],label='sw')
    plt.plot(test[col],label='ms')
    plt.plot((ref.loc[28:,col] + test[col])/2,label='ens')
    plt.title(col)
    plt.legend()
    
#%% submission
cols = ['date','smp_max','smp_min','smp_mean','supply']
submit = pd.concat([target.loc[736:763,cols],test.loc[:,cols]],axis=0)
submit.to_csv('submit/submit_fin_ms.csv',index=False)
