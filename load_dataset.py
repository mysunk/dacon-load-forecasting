# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:33:34 2020

@author: guseh
"""

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

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

del weather_list[1:37] # 0, 37, 38만 남김

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

#%% DFT한 temperature를 피쳐로 추가
# temp = hourly_weather.temp.values
# freq = np.fft.fft(temp)
# M = 10
# N = len(temp)
# freq_filtered = freq.copy()
# freq_filtered[M:-M] = 0
# omega = np.array([2*np.pi/N*i for i in range(N)])
# fn_temp_filtered = lambda t: (1/N)*(np.sum(-freq_filtered.imag*np.sin(omega*t)+freq_filtered.real*np.cos(omega*t)))
# temp_filtered = np.array([fn_temp_filtered(t) for t in range(N)])

# target_dates = pd.date_range('2020-2-1', '2020-3-5',freq='H')
# target_dates = target_dates.delete(-1)
# temp_predict = np.array([fn_temp_filtered(t) for t in range(N, N+len(target_dates))])
# temp_predict = pd.DataFrame(temp_predict, index=target_dates)
# temp_predict = temp_predict.reset_index()
# temp_predict.columns=['datetime', 'temp_predict']

# hourly_weather['temp_s'] = temp_filtered

#%% landtemp를 mean값으로
# hourly_weather['landtemp'] = hourly_weather.loc[:,'landtemp_30cm':'landtemp_20cm'].mean(axis=1)
# hourly_weather.drop(columns=['landtemp_30cm','landtemp_5cm','landtemp_10cm','landtemp_20cm'],inplace=True)

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
    
#%%
temp_pred_list = []
for column in ['temp_max','temp_min','temp_mean']:
    temp = target[column].values
    freq = np.fft.fft(temp)
    M = 10
    N = len(temp)
    freq_filtered = freq.copy()
    freq_filtered[M:-M] = 0
    omega = np.array([2*np.pi/N*i for i in range(N)])
    fn_temp_filtered = lambda t: (1/N)*(np.sum(-freq_filtered.imag*np.sin(omega*t)+freq_filtered.real*np.cos(omega*t)))
    temp_filtered = np.array([fn_temp_filtered(t) for t in range(N)])
    
    target_dates = pd.date_range('2020-2-1', '2020-3-5',freq='H')
    target_dates = target_dates.delete(-1)
    temp_predict = np.array([fn_temp_filtered(t) for t in range(N, N+len(target_dates))])
    temp_predict = pd.DataFrame(temp_predict, index=target_dates)
    temp_predict = temp_predict.reset_index()
    temp_predict.columns=['datetime', column]
    temp_pred_list.append(temp_predict)
    target[column+'_s'] = temp_filtered

temp_pred = pd.concat(temp_pred_list,axis=1)
temp_pred = temp_pred.drop(columns=['datetime'])

#%% 모델 구축 -- supply 예측 모델
def create_model(train, val, label_columns):
    param =  {
        'max_depth':                300,
        'max_features':             10,
        'n_estimators':             1000,
        'random_state' :            0,
       }

    model = RandomForestRegressor(n_jobs=-1,**param)
    # model = RandomForestRegressor(n_jobs=-1)
    model.fit(train[0],train[1])
    y_pred = model.predict(val[0])
    for i, label in enumerate(label_columns):
        plt.figure()
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.plot(np.array(val[1][:,i]), '.-', label='y_val')
        plt.plot(y_pred[:,i], '.-', label='y_pred')
        plt.title(str(future)+'days later'+' of '+label)
        plt.legend()
        plt.show()
        print(f'mae for {label} is {mean_absolute_error(val[1][:,i], y_pred[:,i])}')
        
    return model

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

past = 7 # 최근 7일 정보를 이용하여 n일 후를 예측
past = past-1

x_columns = target.columns[4:]
y_columns = target.columns[9:]
y_columns = y_columns.insert(0,'supply')
supply_models = {}

#%% 7일~34일 후를 예측하는 각각의 모델 구축
for future in range(7, 35):
    train_split = target.shape[0]-past-future-30 # 마지막 30일을 validation set으로 사용
    x_train, y_train = trans(target, 0, train_split, past, future, x_columns, y_columns)
    x_val, y_val = trans(target, train_split, None, past, future, x_columns, y_columns)
    d_train = (x_train,y_train)
    d_val = (x_val,y_val)
    supply_models[future] = create_model(d_train, d_val,y_columns)
    print('==========================================================================')
    
#%% supply 등 예측
x_test = np.array(target.loc[700:, target.columns[4:]])
x_test = x_test.reshape(1,-1)
result_1=[]
for future in range(7, 35):
    result_1.append(supply_models[future].predict(x_test))
result_1 = np.concatenate(result_1,axis=0)

submission = pd.read_csv('data_raw/sample_submission.csv')
submission_bottom_half = submission.loc[28:,:]
submission = submission.loc[:27, :]
test = submission.copy()

test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['dayofweek'] = test['date'].dt.dayofweek

test['supply'] = result_1[:,0] # 이미 있으므로..
for i in range(len(target.columns[9:])):
    test[target.columns[i+9]] = result_1[:,i+1]

#%% 모델 구축 -- smp 예측 모델 -- 시계열 x, cv로 학습시킴
def create_model(x_data, y_data, k=5):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=77)
    
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data.iloc[train_idx], y_data.iloc[train_idx]
        x_val, y_val = x_data.iloc[val_idx], y_data.iloc[val_idx]
    
        param =  {
            'max_depth':                100,
            'max_features':             30,
            'n_estimators':             100,
            'random_state' :            0,
           }

        model = RandomForestRegressor(n_jobs=-1,**param)
        model.fit(x_train,y_train)
        
        plt.rcParams['figure.figsize'] = [12, 4]
        plt.plot(np.array(y_val), '.-', label='y_val')
        plt.plot(model.predict(x_val), '.-', label='y_pred')
        plt.legend()
        plt.show()
        models.append(model)

    return models

x_train = target.iloc[:,4:]
y_train = target.loc[:, ['smp_min', 'smp_max', 'smp_mean' ]]

smp_models = create_model(x_train, y_train, 10)
print('==========================================================================')

#%% smp 예측
x_test = test.loc[:,'supply':]
preds = []
for i in range(10):
    preds.append(smp_models[i].predict(x_test))
pred = np.mean(preds,axis=0)
test.loc[:,['smp_min', 'smp_max', 'smp_mean']] = pred

#%% 결과 제출
submission.loc[:, ['smp_min', 'smp_max', 'smp_mean', 'supply']] = test.loc[:,['smp_min', 'smp_max', 'smp_mean', 'supply']]
submission = pd.concat([submission, submission_bottom_half], axis = 0)
submission.to_csv('submit/submit_baseline.csv', index=False)
