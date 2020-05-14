# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:51:23 2020

@author: guseh
"""


#%% 온도와 전력수급으로 smp 예측하기 -- rf
x_train = target.loc[:,'supply':]
y_train = target.loc[:,'smp_max':'smp_mean']

trials = load_obj('tmp_smp')
param = trials[0]['params']

model = RandomForestRegressor(n_jobs=-1,**param)
model.fit(x_train.values, y_train.values)
x_test = test.loc[:,'supply':]
smp_pred = model.predict(x_test)
test.loc[:,'smp_max':'smp_mean'] = smp_pred

#%% 온도만으로 전력수급 예측하기 -- rf, cv
param =  {
    'max_depth':                300,
    'max_features':             3,
    'n_estimators':             1000,
    'random_state' :            0,
   }

x_train = target.loc[:,'month':]
y_train = target['supply']

model = RandomForestRegressor(n_jobs=-1,**param)
model.fit(x_train.values, y_train.values)

submission = pd.read_csv('data_raw/sample_submission.csv')
submission_bottom_half = submission.loc[28:,:]
submission = submission.loc[:27, :]
test = submission.copy()
test['date'] = pd.to_datetime(test['date'])
# target['year'] = target['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['dayofweek'] = test['date'].dt.dayofweek
test = pd.concat([test, temp_pred],axis=1).copy()
test_x = test.loc[:,'month':]

supply_pred = model.predict(test_x)
test['supply'] = supply_pred

#%% 온도만으로 전력수급 예측하기 -- lgb, cv
trials = load_obj('tmp')
param = trials[0]['params']
nfold = 10
losses = np.zeros((nfold,2)) # 0:train, 1:val
preds = []
models = []
supply_preds = []
kf = KFold(n_splits=nfold, random_state=None, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
    print(i,'th fold training')
    # train test split
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
    model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain, dvalid], num_boost_round=1000,verbose_eval=True,
                             early_stopping_rounds=10)
    train_pred = model.predict(x_train)
    valid_pred = model.predict(x_test)
    losses[i,0]= mean_absolute_error(y_train, train_pred)
    losses[i,1]= mean_absolute_error(y_test, valid_pred)
    
    supply_preds.append(model.predict(test.loc[:,'month':]))

test['supply'] = np.mean(supply_preds,axis=0)

#%% DFT 한 것으로 바꿈
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
    
    target_dates = pd.date_range('2020-2-7', '2020-3-5',freq='D')
    temp_predict = np.array([fn_temp_filtered(t) for t in range(N, N+len(target_dates))])
    temp_predict = pd.DataFrame(temp_predict, index=target_dates)
    temp_predict = temp_predict.reset_index()
    temp_predict.columns=['datetime', column]
    temp_pred_list.append(temp_predict)
    # target[column] = temp_filtered

temp_pred = pd.concat(temp_pred_list,axis=1)
temp_pred = temp_pred.drop(columns=['datetime'])