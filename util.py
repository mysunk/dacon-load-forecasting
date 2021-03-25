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
import pandas as pd

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

def rmsse_lgb(y_pred,train_data, y_hist):
    '''
    y_true: 실제 값 
    y_pred: 예측 값
    y_hist: 과거 값 (public LB는 v1 기간으로 계산, private LB는 v2 기간으로 계산)
    '''
    axis=0

    y_true = train_data.get_label()
    y_true, y_pred, y_hist = np.ravel(y_true), np.ravel(y_pred), np.ravel(y_hist)
    h, n = len(y_true), len(y_hist)
    numerator = np.sum((y_true - y_pred)**2, axis = axis)
    denominator = 1/(n-1)*np.sum((y_hist[1:] - y_hist[:-1])**2, axis = axis)
    msse = 1/h * numerator/denominator
    rmsse = msse ** 0.5
    score = rmsse
    return 'rmsse_modified', score, False


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


def trans(dataset, start_index, end_index, pasts, future, x_columns, y_columns):
    dataset.index = range(dataset.shape[0])
    
    labels = []
    data_agg = []
    start_index_save = start_index
    for j,x_column in enumerate(x_columns):
        start_index = start_index  + pasts[j]
        data = []
        dataset_sub = dataset.loc[:,x_column].copy()
        
        if end_index is None:
            end_index = dataset_sub.shape[0]
        
        for i in range(start_index, end_index-future):
            indices = np.array(dataset_sub.loc[i-pasts[j]:i])
            data.append(indices)
        data = np.array(data)
        data = data.reshape(data.shape[0], -1)
        data = data[max(pasts) - pasts[j]:,:]
        data_agg.append(data)
        start_index = start_index_save
    
    data = np.concatenate(data_agg, axis=1)
    
    for i in range(start_index_save + max(pasts), end_index-future):
        labels.append(np.array(dataset.loc[i+future, y_columns]))
    
    labels = np.array(labels)
    labels = labels.reshape(-1)
    
    return data, labels

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

def load_dataset_v1(features):
    path = 'data_raw/AIFrienz S3_v1'
    
    weather = pd.read_csv(path+'/weather.csv', low_memory=False, dtype={
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
    hourly_smp = pd.read_csv(path+'/hourly_smp.csv')
    target = pd.read_csv(path+'/target.csv')
    
    start = '2018-02-01'
    end = '2020-02-01'
    date_range = pd.date_range(start, end, freq='H')
    hourly_smp['datetime'] = date_range[1:]
    hourly_smp['year'] = hourly_smp['datetime'].dt.year
    hourly_smp['month'] = hourly_smp['datetime'].dt.month
    hourly_smp['day'] = hourly_smp['datetime'].dt.day
    hourly_smp['dayofweek'] = hourly_smp['datetime'].dt.dayofweek
    hourly_smp['hour'] = hourly_smp['datetime'].dt.hour
    
    
    target['date'] = pd.to_datetime(target['date'])
    target['year'] = target['date'].dt.year
    target['month'] = target['date'].dt.month
    target['day'] = target['date'].dt.day
    target['dayofweek'] = target['date'].dt.dayofweek
    
    # rad는 nan값을 0으로 바꿔줌
    a = weather['rad'].values
    a[np.isnan(weather['rad'].values)] = 0
    weather['rad'] = a
    
    
    # weather data 만들기
    # features = ['datetime','area','landtemp_30cm','landtemp_5cm','landtemp_10cm','landtemp_20cm','temp','rad','humid']
    # features = ['datetime','area','temp']
    
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
    else: # 4개 지역
        a = len(features)-1
        for i in range(a):
            hourly_weather[features[1+i]] = hourly_weather.iloc[:,[i+1,i+1+a,i+1+2*a, i+1+3*a]].median(axis=1)
        hourly_weather = hourly_weather.drop(columns = hourly_weather.columns[1:-1*a])
    
    # 결측치 처리 -- 바로 다음값으로 바꿈
    for nums,i in enumerate(hourly_weather.columns):
        if sum(hourly_weather[i].isna()) != 0:
            new_val = hourly_weather.shift(-1)[i][hourly_weather[i].isna()].copy()
            hourly_weather[i][hourly_weather[i].isna()] = new_val
            
    # 결측치 처리 -- linear interpolation
    # hourly_weather = hourly_weather.interpolate()
    
    # hourly를 daily로 바꿈 (concatenate 해서..)
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
    
    # target 만들기
    for i in hourly_weather.columns[1:]:
        target[i+'_max'] = daily_weather.loc[:,i+'_h0':i+'_h23'].max(axis=1)
        target[i+'_min'] = daily_weather.loc[:,i+'_h0':i+'_h23'].min(axis=1)
        target[i+'_mean'] = daily_weather.loc[:,i+'_h0':i+'_h23'].mean(axis=1)
        # target[i+'_std'] = daily_weather.loc[:,i+'_h0':i+'_h23'].std(axis=1)
        
    # hourly smp를 daily로
    hourly_smp.loc[hourly_smp['smp'] < 52,'smp'] = 52
    hourly_smp.loc[hourly_smp['smp'] > 235,'smp'] = 235
    
    start = '2018-02-01'
    end = '2020-01-31'
    daily_smp = pd.DataFrame(columns = ['date'])
    date_range = pd.date_range(start, end, freq='D')
    daily_smp['date'] = date_range
    
    ## 이부분 좀 오래..
    for d in range(daily_smp.shape[0]):
        for h in range(24):
            daily_smp.loc[d, 'smp_h'+str(h)] = hourly_smp['smp'][d*24:d*24+24][d*24+h]
    
    target['smp_max_trunc'] = daily_smp.loc[:,'smp_h0':'smp_h23'].max(axis=1)
    target['smp_min_trunc'] = daily_smp.loc[:,'smp_h0':'smp_h23'].min(axis=1)
    target['smp_mean_trunc'] = daily_smp.loc[:,'smp_h0':'smp_h23'].mean(axis=1)
    return target


def load_dataset_v2(features):
    path = 'data_raw/AIFrienz S3_v2'
    
    weather = pd.read_csv(path+'/weather.csv', low_memory=False, dtype={
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
    hourly_smp = pd.read_csv(path+'/hourly_smp.csv')
    target = pd.read_csv(path+'/target.csv')
    
    start = '2018-02-01'
    end = '2020-05-19'
    date_range = pd.date_range(start, end, freq='H')
    hourly_smp['datetime'] = date_range[1:]
    hourly_smp['year'] = hourly_smp['datetime'].dt.year
    hourly_smp['month'] = hourly_smp['datetime'].dt.month
    hourly_smp['day'] = hourly_smp['datetime'].dt.day
    hourly_smp['dayofweek'] = hourly_smp['datetime'].dt.dayofweek
    hourly_smp['hour'] = hourly_smp['datetime'].dt.hour
    
    
    target['date'] = pd.to_datetime(target['date'])
    target['year'] = target['date'].dt.year
    target['month'] = target['date'].dt.month
    target['day'] = target['date'].dt.day
    target['dayofweek'] = target['date'].dt.dayofweek
    
    # rad는 nan값을 0으로 바꿔줌
    a = weather['rad'].values
    a[np.isnan(weather['rad'].values)] = 0
    weather['rad'] = a
    
    
    # weather data 만들기
    # features = ['datetime','area','landtemp_30cm','landtemp_5cm','landtemp_10cm','landtemp_20cm','temp','rad','humid']
    # features = ['datetime','area','temp']
    
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
    end = '2020-05-18'
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
    else: # 4개 지역
        a = len(features)-1
        for i in range(a):
            hourly_weather[features[1+i]] = hourly_weather.iloc[:,[i+1,i+1+a,i+1+2*a, i+1+3*a]].median(axis=1)
        hourly_weather = hourly_weather.drop(columns = hourly_weather.columns[1:-1*a])
    
    # 결측치 처리 -- 바로 다음값으로 바꿈
    for nums,i in enumerate(hourly_weather.columns):
        if sum(hourly_weather[i].isna()) != 0:
            new_val = hourly_weather.shift(-1)[i][hourly_weather[i].isna()].copy()
            hourly_weather[i][hourly_weather[i].isna()] = new_val
            
    # 결측치 처리 -- linear interpolation
    # hourly_weather = hourly_weather.interpolate()
    
    # hourly를 daily로 바꿈 (concatenate 해서..)
    start = '2018-02-01'
    end = '2020-05-18'
    daily_weather = pd.DataFrame(columns = ['date'])
    date_range = pd.date_range(start, end, freq='D')
    daily_weather['date'] = date_range
    
    ## 이부분 좀 오래..
    for i in range(hourly_weather.shape[1]-1):
        for d in range(daily_weather.shape[0]):
            for h in range(24):
                # 시간별을 feature로 추가
                daily_weather.loc[d, hourly_weather.columns[i+1]+'_h'+str(h)] = hourly_weather[hourly_weather.columns[i+1]][d*24:d*24+24][d*24+h]
    
    # target 만들기
    for i in hourly_weather.columns[1:]:
        target[i+'_max'] = daily_weather.loc[:,i+'_h0':i+'_h23'].max(axis=1)
        target[i+'_min'] = daily_weather.loc[:,i+'_h0':i+'_h23'].min(axis=1)
        target[i+'_mean'] = daily_weather.loc[:,i+'_h0':i+'_h23'].mean(axis=1)
        # target[i+'_std'] = daily_weather.loc[:,i+'_h0':i+'_h23'].std(axis=1)
        
    # hourly smp를 daily로
    hourly_smp.loc[hourly_smp['smp'] < 52,'smp'] = 52
    hourly_smp.loc[hourly_smp['smp'] > 235,'smp'] = 235
    
    start = '2018-02-01'
    end = '2020-05-18'
    daily_smp = pd.DataFrame(columns = ['date'])
    date_range = pd.date_range(start, end, freq='D')
    daily_smp['date'] = date_range
    
    ## 이부분 좀 오래..
    for d in range(daily_smp.shape[0]):
        for h in range(24):
            daily_smp.loc[d, 'smp_h'+str(h)] = hourly_smp['smp'][d*24:d*24+24][d*24+h]
    
    target['smp_max_trunc'] = daily_smp.loc[:,'smp_h0':'smp_h23'].max(axis=1)
    target['smp_min_trunc'] = daily_smp.loc[:,'smp_h0':'smp_h23'].min(axis=1)
    target['smp_mean_trunc'] = daily_smp.loc[:,'smp_h0':'smp_h23'].mean(axis=1)
    return target