# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:36:58 2020

@author: guseh
"""

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from util import *
from functools import partial

weather = pd.read_csv('data_raw/weather_v2.csv', low_memory=False, dtype={
    'area': str,
    'temp_QCFlag': str,
    'prec_QCFlag': str,
    'ws_QCFlag': str,
    'wd_QCFlag': str,
    'humid_QCFlag': str,
    'landP_QCFlag': str,
    'seaP_QCFlag': str,
    'suntime_QCFlag': str,
    'sfctemp_QCFlag': str,
})
hourly_smp = pd.read_csv('data_raw/hourly_smp_v2.csv')
target = pd.read_csv('data_raw/target_v2.csv')

target['date'] = pd.to_datetime(target['date'])
target['year'] = target['date'].dt.year
target['month'] = target['date'].dt.month
target['day'] = target['date'].dt.day
target['dayofweek'] = target['date'].dt.dayofweek

# rad는 nan값을 0으로 바꿔줌
a = weather['rad'].values
a[np.isnan(weather['rad'].values)] = 0
weather['rad'] = a


#%% weather data 만들기
# features = ['datetime','area','landtemp_30cm','landtemp_5cm','landtemp_10cm','landtemp_20cm','temp','rad','humid']
features = ['datetime','area','temp']

weather['datetime'] = pd.to_datetime(weather['datetime'])
weather = weather.loc[:,features].copy()

weather_list = []
for area in weather['area'].unique():
    weather_list.append(weather[weather['area']==area].copy())

del features[1]    
for i, area in enumerate(weather['area'].unique()):
    weather_list[i].drop(['area'], axis=1, inplace=True)
    new_f = []
    for iters, j in enumerate(features):
        if iters==0:
            new_f.append(j)
        else:                
            new_f.append(str(area)+'_' + j)
    weather_list[i].columns = new_f

del weather_list[4:] # 0, 37, 38만 남김
# weather_list = [weather_list[0]]

# hourly
start = '2018-02-01'
end = '2020-01-31'
hourly_weather = pd.DataFrame(columns = ['datetime'])
date_range = pd.date_range(start, end, freq='H')
hourly_weather['datetime'] = date_range

for d in weather_list:
    hourly_weather  = hourly_weather.merge(d, how='outer')

# 같은 feature별로 median 구함
if len(weather_list) == 1:
    a = len(features)-1
    for i in range(a):
        hourly_weather[features[1+i]] = hourly_weather.iloc[:,i+1]
    hourly_weather = hourly_weather.drop(columns = hourly_weather.columns[1:-1])
else: # 3개 지역
    a = len(features)-1
    for i in range(a):
        hourly_weather[features[1+i]] = hourly_weather.iloc[:,[i+1,i+1+a,i+1+2*a, i+1+3*a]].median(axis=1)
    hourly_weather = hourly_weather.drop(columns = hourly_weather.columns[1:-1*a])

# 결측치 처리 -- 바로 다음값으로 바꿈
for nums,i in enumerate(hourly_weather.columns):
    if sum(hourly_weather[i].isna()) != 0:
        new_val = hourly_weather.shift(-1)[i][hourly_weather[i].isna()].copy()
        hourly_weather[i][hourly_weather[i].isna()] = new_val

#%% hourly를 daily로 바꿈 (concatenate 해서..)
start = '2018-02-01'
end = '2020-01-31'
daily_weather = pd.DataFrame(columns = ['date'])
date_range = pd.date_range(start, end, freq='D')
daily_weather['date'] = date_range

## 이부분 좀 오래..
for i in range(hourly_weather.shape[1]-1):
    for d in range(daily_weather.shape[0]):
        for h in range(24):
            # 시간별을 feature로 추가
            daily_weather.loc[d, hourly_weather.columns[i+1]+'_h'+str(h)] = hourly_weather[hourly_weather.columns[i+1]][d*24:d*24+24][d*24+h]

#%% target 만들기
for i in hourly_weather.columns[1:]:
    target[i+'_max'] = daily_weather.loc[:,i+'_h0':i+'_h23'].max(axis=1)
    target[i+'_min'] = daily_weather.loc[:,i+'_h0':i+'_h23'].min(axis=1)
    target[i+'_mean'] = daily_weather.loc[:,i+'_h0':i+'_h23'].mean(axis=1)
    # target[i+'_std'] = daily_weather.loc[:,i+'_h0':i+'_h23'].std(axis=1)


#%% target 저장
# target.to_csv('data/target.csv',index=False)
# target.to_csv('data/target_year_included.csv',index=False)
target = pd.read_csv('data/target.csv')
#%% 28일간 예측
past = 28
past = past -1

submission = pd.read_csv('data_raw/sample_submission.csv')
submission_bottom_half = submission.loc[28:,:]
submission = submission.loc[:27, :]
test = submission.copy()
test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['dayofweek'] = test['date'].dt.dayofweek
# test = pd.concat([test, temp_pred],axis=1).copy() # DFT로 예측한 temp값

x_test = np.array(target.loc[target.shape[0]-past-1:, target.columns[4:]])
x_test = x_test.reshape(1,-1)

#%% 온도만으로 전력수급 예측하기 -- lgb, time series -- loss 확인용
# x_columns, y_columns =  target.columns[4:], ['supply','temp_mean', 'temp_min','temp_max']
# losses = np.zeros((28, len(y_columns)))
# random_states = []
# for random_state in range(10):
#     for i, y_column in enumerate(y_columns):
#         if y_column == 'supply':
#             trials = load_obj('0513/result1_supply')
#             param = trials[0]['params']
#         else:
#             trials = load_obj('0513/result1_'+y_column)
#             param = trials[0]['params']
#         # param = {
#         # 'metric': 'l1',
#         # 'seed':7777
#         # }
        
#         param['random_state'] = random_state
#         # model 만들기
#         for future in range(7, 35):
#         # for future in [7]:
#             train_split = target.shape[0]-past-future-28 # 마지막 30일을 validation set으로 사용
#             x_train, y_train = trans(target, 0, train_split, past, future, x_columns, y_column)
#             x_val, y_val = trans(target, train_split, None, past, future, x_columns, y_column)
#             y_train = np.ravel(y_train)
#             y_val = np.ravel(y_val)
#             d_train = lgb.Dataset(x_train,label=y_train)
#             d_val = lgb.Dataset(x_val,label=y_val)
#             model = lgb.train(param, train_set = d_train,  
#                           valid_sets=[d_train, d_val],num_boost_round=1000,verbose_eval=False,
#                                   early_stopping_rounds=10)
            
#             y_pred = model.predict(x_val)
#             loss = model.best_score['valid_1']['l1']
#             losses[future-7,i] = loss
#         break
#     print(f'for {random_state}, {np.mean(losses, axis=0)}')
        
    
#%% supply 예측
x_columns, y_columns =  target.columns[4:], ['supply','temp_mean', 'temp_min','temp_max']
for i, y_column in enumerate(y_columns):
    pred_all = []
    for random_state in [0]:
        trials = load_obj('0513/result1_'+y_column)
        param = trials[0]['params']

        # param = {
        # 'metric': 'l1',
        # 'seed':777
        # }
        models = []
        param['random_state'] = random_state
        # model 만들기
        for future in range(7, 35):
            train_split = target.shape[0]-past-future-30 # 마지막 30일을 validation set으로 사용
            x_train, y_train = trans(target, 0, train_split, past, future, x_columns, y_column)
            x_val, y_val = trans(target, train_split, None, past, future, x_columns, y_column)
            y_train = np.ravel(y_train)
            y_val = np.ravel(y_val)
            d_train = lgb.Dataset(x_train,label=y_train)
            d_val = lgb.Dataset(x_val,label=y_val)
            model = lgb.train(param, train_set = d_train,  
                          valid_sets=[d_train, d_val],num_boost_round=1000,verbose_eval=False,
                                  early_stopping_rounds=10)
            models.append(model)
        
        preds = []
        for future in range(7, 35):
            preds.append(models[future-7].predict(x_test))
        preds = np.concatenate(preds,axis=0)
        pred_all.append(preds)
    test[y_column] = np.mean(pred_all, axis=0)
    
#%% 온도와 전력수급으로 smp 예측하기
train = target.loc[:,'supply':]
y_columns = ['smp_max','smp_min','smp_mean']
test_x = test.loc[:,'supply':]

nfolds = 5
losses = np.zeros((nfolds,len(y_columns)))
random_states = [0]
for j,y_column in enumerate(y_columns):
    pred_all = []
    for state in random_states:
        # trials = load_obj('0517/'+y_column)
        # trials = load_obj('0515/'+y_column)
        # params = trials[0]['params']
        
        params = {
            'seed':777,
            'metric':'mse'
        }
        params['random_state']=state
        # params['metric']='None'

        train_label = target.loc[:,y_column]
        preds = []
        kf = KFold(n_splits=nfolds,random_state=None, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
            x_train = train.iloc[train_index]
            y_train = train_label.iloc[train_index]
            x_test = train.iloc[test_index]
            y_test = train_label.iloc[test_index]
            
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_test, label=y_test)
            # rmmse_obj = partial(rmsse_lgb, y_hist = train_label)
            # model = lgb.train(params, train_set = dtrain,  
            #                   valid_sets=[dtrain, dvalid],num_boost_round=1000,verbose_eval=True,feval=rmmse_obj,
            #                           early_stopping_rounds=10)
            # loss = model.best_score['valid_1']['rmsse_modified']
            model = lgb.train(params, train_set = dtrain,  
                              valid_sets=[dtrain, dvalid],num_boost_round=1000,verbose_eval=False,
                                      early_stopping_rounds=10)
            loss = model.best_score['valid_1']['l2']
            
            losses[i,j] = loss
            
            pred = model.predict(test_x)
            preds.append(pred)
        pred_all.append(np.mean(preds,axis=0))
    test[y_column] = np.mean(pred_all,axis=0)


#%% 최종 제출
submission.loc[:, ['smp_min', 'smp_max', 'smp_mean', 'supply']] = test.loc[:,['smp_min', 'smp_max', 'smp_mean', 'supply']]
submission = pd.concat([submission, submission_bottom_half], axis = 0)
submission.to_csv('submit/tmp.csv', index=False)