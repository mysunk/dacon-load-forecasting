# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:46:56 2020

@author: guseh
"""


import pandas as pd
from util import *
result_1 = pd.read_csv('submit/submit_20.csv')
result_2 = pd.read_csv('submit/submit_21.csv')
result_3 = pd.read_csv('submit/submit_22.csv')

ref = pd.read_csv('submit/submit_5.csv')

#%% ensemble
ensemble = result_1.values[:,1:] * 0.6 + result_2.values[:,1:] * 0.3 + result_3.values[:,1:] * 0.1
submission = pd.read_csv('data_raw/sample_submission.csv')
submission.iloc[:,1:] = ensemble
submission.to_csv('submit/submit_19.csv', index=False)

#%% 결과 확인
target = pd.read_csv('data/target.csv')
submit_1 = pd.read_csv('submit/submit_21.csv')
y_true = submit_1.loc[:28,'smp_max':'supply'].values
submit_2 = pd.read_csv('submit/sub_baseline_v4.csv')
y_pred = submit_2.loc[:28,'smp_max':'supply'].values
print(rmsse(y_true, y_pred, target.loc[:,'smp_max':'supply'].values, axis = 0, weight = [0.1, 0.1, 0.2, 0.6]))