# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:20:01 2020

@author: guseh
"""
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from util import *
#%% SMP 분석
# hourly 데이터
tmp = np.reshape(hourly_smp.smp.values,[-1,24])
plt.plot(tmp[:100,:])

# smp label 데이터
plt.plot(label.smp_mean.values)

#%% 수요 분석
date = label.date.values
tmp = label.supply.values
plt.figure()
plt.plot(range(730), tmp)

#%% 수요의 seasonal decomposition
label = pd.read_csv('data_raw/target_v1.csv')

seasonal = seasonal_decompose(label.supply.values,period = 24)
trend = seasonal.trend
season = seasonal.seasonal
residue = seasonal.resid
plt.figure()
plt.subplot(4,1,1)
plt.plot(label.supply.values)
plt.subplot(4,1,2)
plt.plot(trend)
plt.subplot(4,1,3)
plt.plot(season)
plt.subplot(4,1,4)
plt.plot(residue)

#%% ACF 분석
demand = label.loc[:,['date','supply']]
plot_acf(demand)
plot_pacf(demand)
plt.show()

#%% AR을 이용한 전력 수요 train
model = ARIMA(demand.supply.values, order = (1,1,1))
model_fit = model.fit(trend='c',full_output=True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()
#%% DFT로 noise 제거
freq = np.fft.fft(demand.supply.values)
M = 15
N = len(demand.supply.values)
freq_filtered = freq.copy()
freq_filtered[M:-M] = 0
omega = np.array([2*np.pi/N*i for i in range(N)])
fn_filtered = lambda t: (1/N)*(np.sum(-freq_filtered.imag*np.sin(omega*t)+freq_filtered.real*np.cos(omega*t)))
supply_filtered = np.array([fn_filtered(t) for t in range(N)])
plt.plot(demand.supply.values)
plt.plot(supply_filtered)

supply_fore = np.array([fn_filtered(t) for t in range(N,N+35)])

#%% ARMA
model = ARIMA(demand.supply.values, order = (1,0,1))
model_fit = model.fit(trend='c',full_output=True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()

#%% pred
fore = model_fit.forecast(steps=35)
plt.plot(range(730),demand.supply.values)
plt.plot(range(730,765),supply_fore)

#%% 일자별 온도의 min값
date = pd.date_range('2018-2-1', '2020-2-1',freq='H')
tmp = weather[weather.area.values == 184]
tmp = tmp.loc[:,['datetime','temp','humid']]
tmp_u = tmp.iloc[0:1,:]
tmp_l = tmp.iloc[-1:,:]
a = pd.concat([tmp_u, tmp, tmp_l],axis=0, ignore_index=True)
a.loc[:,'datetime'] = date[0:-1]

wth = a.values[:,1:]


#%% 결측치가 작은 지역 고르기 -- 0, 37, 38
for i in range(39):
    print(f'======{i}th area=====')
    print(weather_list[i].isna().sum())
    
    
#%% hourly_weather 보기
temp = hourly_weather.temp.values
plt.plot(temp)
seasonal = seasonal_decompose(temp,period = 7*24*4*6)
trend = seasonal.trend
season = seasonal.seasonal
residue = seasonal.resid

plt.figure()
plt.plot(temp)
plt.plot(temp-season)

#%% DFT
freq = np.fft.fft(temp)
M = 30
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
temp_predict.columns=['datetime', 'temp_predict']

#%% plot
plt.figure()
plt.plot(hourly_weather['datetime'],temp)
plt.plot(hourly_weather['datetime'], temp_filtered, color='tomato', lw=4, label='Filtered')
plt.plot(temp_predict['datetime'], temp_predict['temp_predict'], color='tomato', lw=4, label='Filtered')
plt.legend()


#%%
plt.figure()
plt.plot(target['date'],target['smp_mean'])
plt.plot(test['date'],test['smp_mean'])



#%%
plt.plot(target['temp_mean'])
plt.plot(temp_filtered)
plt.plot(temp_pred['temp_mean'])

#%% base
trials = load_obj('0515_base/supply_21')
trials = load_obj('0515_base/supply_28')
trials = load_obj('0515_base/supply_35')
#%% day
trials = load_obj('0515_day/supply_4')
trials = load_obj('0515_day/supply_5')
trials = load_obj('0515_day/supply_6')
#%% w
trials = load_obj('0515_nonworkday/supply_4')
trials = load_obj('0515_nonworkday/supply_8')
trials = load_obj('0515_nonworkday/supply_12')
#%% n/w
trials = load_obj('0515_workday/supply_5')
trials = load_obj('0515_workday/supply_10')
trials = load_obj('0515_workday/supply_15')

#%%
plt.plot(target['temp_max'], target['supply'],'.')

#%%
trials = load_obj('0516/smp_max')
trials = load_obj('0516/smp_min')
trials = load_obj('0516/smp_mean')

trials = load_obj('0513/result1_temp_min')

#%%
trials = load_obj('0516_base/supply_28') # 4.87
trials = load_obj('0516_workday/supply_28') # 5.21
