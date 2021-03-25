# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:02:31 2020

@author: guseh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% smp 불러오기
target = pd.read_csv('D:/GITHUB/dacon_load/data_raw/AIFrienz S3_v2/target.csv')
target_sub = pd.DataFrame(columns = ['date'])
target_sub['date'] = pd.to_datetime(target['date'])
target_sub['smp_mean'] = target['smp_mean']
target_sub['smp_min'] = target['smp_min']
target_sub['smp_max'] = target['smp_max']


#%% bin 만들기
start = '2018-02-01'
end = '2020-05-18'
tmp = pd.DataFrame(columns = ['date'])
date_range = pd.date_range(start, end, freq='D')
tmp['date'] = date_range

#%% 유가, 환율 불러오기
brent = pd.read_csv('../etc/oil_price_brent.csv')
du = pd.read_csv('../etc/oil_price_du.csv')
wti = pd.read_csv('../etc/oil_price_wti.csv')
lng = pd.read_csv('../etc/oil_price_lng.csv')
gold = pd.read_csv('../etc/gold.csv', dtype={'usd':float, 'gold':float})

# for i in range(len(gold)):
#     try:
#         # gold.loc[i,'gold'] = gold.loc[i,'gold'].replace(',','')
#         gold.loc[i,'usd'] = gold.loc[i,'usd'].replace(',','')
#     except:
#        pass

gold = gold.iloc[3555:-4,:].copy()
gold['date'] = pd.to_datetime(gold['date'])

brent = brent.iloc[4:592,1:].copy()
brent = brent[::-1].reset_index(drop=True)
brent['date'] = pd.to_datetime(brent['date'])

du = du.iloc[4:581,1:].copy()
du = du[::-1].reset_index(drop=True)
du['date'] = pd.to_datetime(du['date'])

wti = wti.iloc[4:582,1:].copy()
wti = wti[::-1].reset_index(drop=True)
wti['date'] = pd.to_datetime(wti['date'])

lng = lng.iloc[4:581,1:].copy()
lng = lng[::-1].reset_index(drop=True)
lng['date'] = pd.to_datetime(lng['date'])

brent = tmp.merge(brent.copy(), how='outer')
du = tmp.merge(du.copy(), how='outer')
wti = tmp.merge(wti.copy(), how='outer')
lng = tmp.merge(lng.copy(), how='outer')
gold = tmp.merge(gold.copy(), how='outer')

oil = tmp.merge(brent.copy(), how='outer')
oil = oil.merge(du.copy(), how='outer')
oil = oil.merge(wti.copy(), how='outer')
oil = oil.merge(lng.copy(), how='outer')
oil = oil.merge(gold.copy(), how='outer')

oil = oil.merge(target_sub.copy(), how='outer')
oil = oil.fillna(method='pad')

oil['supply'] = target['supply']
oil.to_csv('data/target_oil.csv',index=False)

#%% 분석
# oil['gold'] = oil['gold']**(-1)*1000
# oil['usd'] = oil['usd']**(-1)*1000
oil['lng_won'] = oil['lng'] * oil['usd']
oil['gold_won'] = oil['gold'] * oil['usd']
import seaborn as sns
sns.heatmap(oil.corr(), annot=True)
#%% cross correlation
from scipy.signal import correlate

# lag = np.argmax(correlate(oil['lng_won'], oil['smp_mean']))

num = 30
lag = np.roll(oil['gold'] , shift=--1*num)
print(np.corrcoef(oil['smp_mean'],oil['gold']))
print(np.corrcoef(oil.loc[num:,'smp_mean'],lag[num:]))

plt.plot(lag[num:], oil.loc[num:,'smp_mean'],'.')

plt.plot(oil.loc[num:,'gold'], oil.loc[num:,'smp_mean'],'.')
plt.plot(lag)
plt.plot(oil.loc[:,'gold'])
#%%
plt.plot(oil['lng'],oil['smp_mean'],'.')
