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

weather = pd.read_csv('data_raw/weather_v1.csv', low_memory=False, dtype={
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
hourly_smp = pd.read_csv('data_raw/hourly_smp_v1.csv')
target = pd.read_csv('data_raw/target_v1.csv')

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
# target.to_csv('data/target.csv',index = False)

#%% 28일간 예측
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


#%% w/n으로 나누기 -- loss보기
# x_columns, y_column =  target.columns[4:], 'supply'
# past_w = 9-1 # 5개전
# past_n = 26-1 # 2개전

# val_num_w = 28
# val_num_n = 28

# losses_w = []
# losses_n = []
# # param = {
# #     'metric': 'l1',
# #     'seed':777
# #     }

# workday = [0,1,2,3,4]
# trials = load_obj('0517_com/w')
# param = trials[0]['params']

# for j in range(5,25): # 20개
#     # w
#     target_sub = target.loc[target['dayofweek']%7<5 ,:].copy().reset_index(drop=True)
#     train_split = target_sub.shape[0]-past_w-j-val_num_w # 마지막 n개를 validation set으로 사용
#     x_train, y_train = trans(target_sub, 0, train_split, past_w, j, x_columns, y_column)
#     x_val, y_val = trans(target_sub, train_split, None, past_w, j, x_columns, y_column)
#     y_train = np.ravel(y_train)
#     y_val = np.ravel(y_val)
#     d_train = lgb.Dataset(x_train,label=y_train)
#     d_val = lgb.Dataset(x_val,label=y_val)
#     model = lgb.train(param, train_set = d_train,  
#                   valid_sets=[d_train, d_val],num_boost_round=1000,verbose_eval=False,
#                           early_stopping_rounds=10)
#     loss = model.best_score['valid_1']['l1']
#     losses_w.append(loss)

# trials = load_obj('0517_com/nw')
# param = trials[0]['params']
# for j in range(1,9): # 8개
#     # NW
#     target_sub = target.loc[target['dayofweek']%7>=5,:].copy().reset_index(drop=True)
#     train_split = target_sub.shape[0]-past_n-j-val_num_n # 마지막 n개를 validation set으로 사용
#     x_train, y_train = trans(target_sub, 0, train_split, past_n, j, x_columns, y_column)
#     x_val, y_val = trans(target_sub, train_split, None, past_n, j, x_columns, y_column)
#     y_train = np.ravel(y_train)
#     y_val = np.ravel(y_val)
#     d_train = lgb.Dataset(x_train,label=y_train)
#     d_val = lgb.Dataset(x_val,label=y_val)
#     model = lgb.train(param, train_set = d_train,  
#                   valid_sets=[d_train, d_val],num_boost_round=1000,verbose_eval=False,
#                           early_stopping_rounds=10)
#     loss = model.best_score['valid_1']['l1']
#     losses_n.append(loss)
    
#%% w/n으로 나누기 -- supply 예측
x_columns, y_column =  target.columns[4:], 'supply'
target_sub_w = target.loc[target['dayofweek']%7<5 ,:].copy().reset_index(drop=True)
target_sub_n = target.loc[target['dayofweek']%7<5 ,:].copy().reset_index(drop=True)

past_w = 9-1 
past_n = 26-1

x_test_w = np.array(target_sub_w.loc[target_sub_w.shape[0]-past_w-1:, target_sub_w.columns[4:]])
x_test_w = x_test_w.reshape(1,-1)


x_test_n = np.array(target_sub_n.loc[target_sub_n.shape[0]-past_n-1:, target_sub_n.columns[4:]])
x_test_n = x_test_n.reshape(1,-1)
        

preds_w = []
preds_n = []

# param = {
#     'metric': 'l1',
#     'seed':777
#     }
workday = [0,1,2,3,4]
trials = load_obj('0517_com/w')
param = trials[0]['params']

for j in range(5,25): # 20개
    # w
    target_sub = target.loc[target['dayofweek']%7<5 ,:].copy().reset_index(drop=True)
    x_train, y_train = trans(target_sub, 0, None, past_w, j, x_columns, y_column)
    y_train = np.ravel(y_train)
    d_train = lgb.Dataset(x_train,label=y_train)
    model = lgb.train(param, train_set = d_train,  
                  valid_sets=[d_train],num_boost_round=1000,verbose_eval=False,
                         early_stopping_rounds=10)
    preds_w.append(model.predict(x_test_w))

trials = load_obj('0517_com/nw')
param = trials[0]['params']
for j in range(1,9): # 8개
    # NW
    target_sub = target.loc[target['dayofweek']%7>=5,:].copy().reset_index(drop=True)
    x_train, y_train = trans(target_sub, 0, None, past_n, j, x_columns, y_column)
    y_train = np.ravel(y_train)
    d_train = lgb.Dataset(x_train,label=y_train)
    model = lgb.train(param, train_set = d_train,  
                  valid_sets=[d_train],num_boost_round=1000,verbose_eval=False,
                         early_stopping_rounds=10)
    preds_n.append(model.predict(x_test_n))

preds_w = np.ravel(preds_w)
preds_n = np.ravel(preds_n)
y_pred = []
count_w, count_n = 0,0
for i in range(4,32):
    if i%7 <5:
        y_pred.append(preds_w[count_w])
        count_w = count_w+1
    else:
        y_pred.append(preds_n[count_n])
        count_n = count_n+1

test['supply'] = y_pred

#%% 온도 예측 -- loss 보기
# past = 30
# past = past -1

# x_columns, y_columns =  target.columns[4:], ['temp_mean', 'temp_min','temp_max']
# x_test = np.array(target.loc[target.shape[0]-past-1:, target.columns[4:]])
# x_test = x_test.reshape(1,-1)

# losses = np.zeros((28, len(y_columns)))
# for i, y_column in enumerate(y_columns):
#     trials = load_obj('0513/result1_'+y_column)
#     param = trials[0]['params']
#     models = []
    
#     # model 만들기
#     for future in range(7, 35):
#         train_split = target.shape[0]-past-future-30 # 마지막 30일을 validation set으로 사용
#         x_train, y_train = trans(target, 0, train_split, past, future, x_columns, y_column)
#         x_val, y_val = trans(target, train_split, None, past, future, x_columns, y_column)
#         y_train = np.ravel(y_train)
#         y_val = np.ravel(y_val)
#         d_train = lgb.Dataset(x_train,label=y_train)
#         d_val = lgb.Dataset(x_val,label=y_val)
#         model = lgb.train(param, train_set = d_train,  
#                       valid_sets=[d_train, d_val],num_boost_round=1000,verbose_eval=False,
#                              early_stopping_rounds=10)
#         loss = model.best_score['valid_1']['l1']
#         losses[future-7,i] = loss
#         models.append(model)
    
#     preds = []
#     for future in range(7, 35):
#         preds.append(models[future-7].predict(x_test))
#     preds = np.concatenate(preds,axis=0)
    
#     test[y_column] = preds
    
#%% 온도 예측 -- 예측
past = 28
past = past -1

x_columns, y_columns =  target.columns[4:], ['temp_mean', 'temp_min','temp_max']
x_test = np.array(target.loc[target.shape[0]-past-1:, target.columns[4:]])
x_test = x_test.reshape(1,-1)

for i, y_column in enumerate(y_columns):
    trials = load_obj('0513/result1_'+y_column)
    param = trials[0]['params']
    models = []
    
    # model 만들기
    for future in range(7, 35):
        x_train, y_train = trans(target, 0, None, past, future, x_columns, y_column)
        y_train = np.ravel(y_train)
        d_train = lgb.Dataset(x_train,label=y_train)
        model = lgb.train(param, train_set = d_train,  
                      valid_sets=[d_train],num_boost_round=1000,verbose_eval=False,
                             early_stopping_rounds=10)
        models.append(model)
    
    preds = []
    for future in range(7, 35):
        preds.append(models[future-7].predict(x_test))
    preds = np.concatenate(preds,axis=0)
    
    test[y_column] = preds


#%% 온도와 전력수급으로 smp 예측하기 -- lgb
train = target.loc[:,'supply':]
y_columns = ['smp_max','smp_min','smp_mean']
test_x = test.loc[:,'supply':]
params = {
    'metric': 'mse',
    'seed':777
    }

nfolds = 10
losses = np.zeros((nfolds,len(y_columns)))
random_states = range(10)
for j,y_column in enumerate(y_columns):
    pred_all = []
    for state in random_states:
        train_label = target.loc[:,y_column]
        preds = []
        kf = KFold(n_splits=nfolds,random_state=None, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
            if isinstance(train, (np.ndarray, np.generic) ): # if numpy array
                x_train = train[train_index]
                y_train = train_label[train_index]
                x_test = train[test_index]
                y_test = train_label[test_index]
            else: # if dataframe
                x_train = train.iloc[train_index]
                y_train = train_label.iloc[train_index]
                x_test = train.iloc[test_index]
                y_test = train_label.iloc[test_index]
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_test, label=y_test)
            model = lgb.train(params, train_set = dtrain,  
                              valid_sets=[dtrain, dvalid],num_boost_round=1000,verbose_eval=False,
                                     early_stopping_rounds=10)
            losses[i,j] = model.best_score['valid_1']['l2']
            pred = model.predict(test_x)
            preds.append(pred)
        pred_all.append(np.mean(preds,axis=0))
    test[y_column] = np.mean(pred_all,axis=0)
    
#%% 최종 제출
submission.loc[:, ['smp_min', 'smp_max', 'smp_mean', 'supply']] = test.loc[:,['smp_min', 'smp_max', 'smp_mean', 'supply']]
submission = pd.concat([submission, submission_bottom_half], axis = 0)
submission.to_csv('submit/submit_15_wnw.csv', index=False)
