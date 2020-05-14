# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:46:55 2020

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
target['dayinweek'] = target['day'] % 7

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

del weather_list[1:37] # 0, 37, 38만 남김
# weather_list = weather_list[0]

# hourly
start = '2018-02-01'
end = '2020-01-31'
hourly_weather = pd.DataFrame(columns = ['datetime'])
date_range = pd.date_range(start, end, freq='H')
hourly_weather['datetime'] = date_range

for d in weather_list:
    hourly_weather  = hourly_weather.merge(d, how='outer')

# 같은 feature별로 median 구함
a = len(features)-1
for i in range(a):
    hourly_weather[features[1+i]] = hourly_weather.iloc[:,[i+1,i+1+a,i+1+2*a]].median(axis=1)
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
    
#%% 같은요일끼리 묶기
x_columns, y_columns =  target.columns[4:], ['temp_mean', 'temp_min','temp_max','supply']
past = 4-1 # n주 전 같은 요일 
targets = []
model_all = []
# losses = np.zeros((4,7))
losses = []
y_column = 'supply'
param = {
    'metric': 'l1',
    'seed':777
    }
for j in range(2,6): # n주 뒤
    models = []
    for i in range(7): # n요일
        target_sub = target.loc[target['dayinweek'] == i,:].copy()
        train_split = target_sub.shape[0]-past-j-7 # 마지막 7개를 validation set으로 사
        x_train, y_train = trans(target_sub, 0, train_split, past, j, x_columns, y_column)
        x_val, y_val = trans(target_sub, train_split, None, past, j, x_columns, y_column)
        y_train = np.ravel(y_train)
        y_val = np.ravel(y_val)
        d_train = lgb.Dataset(x_train,label=y_train)
        d_val = lgb.Dataset(x_val,label=y_val)
        model = lgb.train(param, train_set = d_train,  
                      valid_sets=[d_train, d_val],num_boost_round=1000,verbose_eval=False,
                             early_stopping_rounds=10)
        loss = model.best_score['valid_1']['l1']
        # losses[j-2,i] = loss
        losses.append(loss)
        models.append(model)
    model_all.append(models)
