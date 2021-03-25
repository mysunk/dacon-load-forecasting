"""
Created on Mon Mar  2 22:44:55 2020

@author: guseh
"""
# packages
import argparse
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from util import *
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from functools import partial
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
# models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Tuning_model(object):
    
    def __init__(self):
        self.random_state = 0
        self.space = {}
    
    # parameter setting
    def lgb_space(self):
        # LightGBM parameters
        self.space = {
            'learning_rate':            hp.uniform('learning_rate',    0.01, 0.1),
            'max_depth':                -1,
            'num_leaves':               hp.quniform('num_leaves',       5, 200, 1), 
            'min_data_in_leaf':		    hp.quniform('min_data_in_leaf',	10, 200, 1),	# overfitting 안되려면 높은 값
            'reg_alpha':                hp.uniform('reg_alpha',0,1),
            'reg_lambda':               hp.uniform('reg_lambda',0, 1),
            'colsample_bytree':         hp.uniform('colsample_bytree', 0, 1.0),
            'colsample_bynode':		    hp.uniform('colsample_bynode',0,1.0),
            'bagging_freq':			    hp.quniform('bagging_freq',	1,20,1),
            'tree_learner':			    hp.choice('tree_learner',	['serial','feature','data','voting']),
            'subsample':                hp.uniform('subsample', 0, 1.0),
            'boosting':			        hp.choice('boosting', ['gbdt']),
            'max_bin':			        hp.quniform('max_bin',		100,300,1), # overfitting 안되려면 낮은 값
            "min_sum_hessian_in_leaf": hp.uniform('min_sum_hessian_in_leaf',       0, 0.1), 
            'random_state':             self.random_state,
            'n_jobs':                   -1,
            'metrics':                  'l1'
        }
            
    # optimize
    def process(self, clf_name, train_set, past, trials, algo, max_evals):
        fn = getattr(self, clf_name+'_val')
        space = getattr(self, clf_name+'_space')
        space()
        fmin_objective = partial(fn, train_set=train_set,past=past)
        try:
            result = fmin(fn=fmin_objective, space=self.space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials
    
    # objective function
    def lgb_val(self, params, train_set, past): 
        params = make_param_int(params, ['max_depth','num_leaves','min_data_in_leaf',
                                     'bagging_freq','max_bin'])
        past = past - 1
        
        # params = {
        #     'metric': 'mae',
        #     'seed':777
        #     }
        trials = load_obj('0513/result1_supply')
        params = trials[0]['params']
        
        losses = []
        for future in range(7, 35):
            train_split = train_set.shape[0]-past-future-28 # 마지막 30일을 validation set으로 사용
            x_train, y_train = trans(train_set, 0, train_split, past, future, x_columns, y_columns)
            x_val, y_val = trans(train_set, train_split, None, past, future, x_columns, y_columns)
            y_train = np.ravel(y_train)
            y_val = np.ravel(y_val)
            d_train = lgb.Dataset(x_train,label=y_train)
            d_val = lgb.Dataset(x_val,label=y_val)
            model = lgb.train(params, train_set = d_train,  
                          valid_sets=[d_train, d_val],num_boost_round=1000,verbose_eval=False,
                                 early_stopping_rounds=10)
            loss = model.best_score['valid_1']['l1']
            losses.append(loss)
            
        return {'loss': np.mean(losses,axis=0),'params':params ,'status': STATUS_OK}
    
if __name__ == '__main__':
    
    # load config
    parser = argparse.ArgumentParser(description='Dacon temperature regression',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='lgb', choices=['lgb', 'eln', 'rf','svr'])
    parser.add_argument('--max_evals', default=10,type=int)
    parser.add_argument('--save_file', default='tmp')
    parser.add_argument('--past', default=28,type=int)
    parser.add_argument('--label', default='supply')
    args = parser.parse_args()
    
    # load dataset
    target = pd.read_csv('data/target.csv')
    label = args.label
    x_columns, y_columns =  target.columns[4:], [label]
    
    # main
    clf = args.method
    bayes_trials = Trials()
    obj = Tuning_model()
    tuning_algo = tpe.suggest
    # tuning_algo = tpe.rand.suggest # -- random search
    obj.process(args.method, target, args.past, 
                           bayes_trials, tuning_algo, args.max_evals)
    
    # save trial
    save_obj(bayes_trials.results,args.save_file)