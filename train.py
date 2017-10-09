# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import gc

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


## STEP0: do setting
class Settings(Enum):
    global train06_path
    global train07_path
    global properties06_path
    global properties07_path
    global submission_path
    global XGB_WEIGHT
    global LGB_WEIGHT
    global USE_VALID_DATA
    global VALID_DATA_RATIO
    global TUNE_PARAMS
    
    train06_path      = 'C:/data/kaggle/zillow_price/train_2016_v2.csv'
    train07_path      = 'C:/data/kaggle/zillow_price/train_2017.csv'
    properties06_path = 'C:/data/kaggle/zillow_price/properties_2016.csv'
    properties07_path = 'C:/data/kaggle/zillow_price/properties_2017.csv'
    submission_path   = 'C:/data/kaggle/zillow_price/sample_submission.csv'
    XGB_WEIGHT        = 0.5
    LGB_WEIGHT        = 1 - XGB_WEIGHT
    USE_VALID_DATA    = False
    VALID_DATA_RATIO  = 0.2
    TUNE_PARAMS       = False
    
    def __str__(self):
        return self.value
        
    
## STEP1: process data
def analyze(df):
    print('\nAnalyzing ...')
    
    # show logerror
    if(1):
        plt.figure(figsize=(8,6))
        plt.scatter(range(df.shape[0]), np.sort(df.logerror.values))
        plt.xlabel('index', fontsize=12)
        plt.ylabel('logerror', fontsize=12)
        plt.show()
        
    # show other features in correlation with logerror
    if(1):
        for feature in df:
            if df[feature].dtype == "float64" or "int64":
                if feature != 'transactiondate':
                    fig, ax = plt.subplots()
                    ax.scatter(x = df[feature], y = df['logerror'])
                    plt.ylabel('logerror', fontsize=13)
                    plt.xlabel(feature, fontsize=13)
                    plt.show()  
                    
    # show correlation
    if(1):
        xgb_params = {
            'eta': 0.05,
            'max_depth': 8,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'silent': 1,
            'seed' : 0
        }
        
        train_y = df.logerror.values
        train_x = df.drop(["transactiondate", "logerror"], axis=1)
        
        dtrain = xgb.DMatrix(train_x, train_y, feature_names=train_x.columns.values)
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
        
        featureImportance = model.get_fscore()
        features = pd.DataFrame()
        features['features'] = featureImportance.keys()
        features['importance'] = featureImportance.values()
        features.sort_values(by=['importance'],ascending=False,inplace=True)
        fig,ax= plt.subplots()
        fig.set_size_inches(10,20)
        plt.xticks(rotation=90)
        sns.barplot(data=features,x="importance",y="features",ax=ax,orient="h",color="#34495e")      
        plt.show()
                    
def fill_NA(df):
    print('\nFilling NA ...')
    
    na_ratio = ((df.isnull().sum() / len(df)) * 100).sort_values(ascending=False)
    print('NA ratio: ')
    print(na_ratio) 
    
    df['hashottuborspa'] = df['hashottuborspa'].fillna("FALSE")
    df['fireplaceflag'] = df['fireplaceflag'].fillna("FALSE")
    
    for feature in df:
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna("None")
        else:
            df[feature] = df[feature].fillna(0)
    
def encode_features(df):
    print('\nEncoding features ...')
    
    for feature in df:
        if df[feature].dtype == 'object':
            print('Encoding ', feature)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[feature].values))
            df[feature] = lbl.transform(list(df[feature].values))
    
def add_features(prop_df):
    print('\nAdding features ...')
    
    zip_count = prop_df['regionidzip'].value_counts().to_dict()
    prop_df['N-zip_count'] = prop_df['regionidzip'].map(zip_count)
    
    city_count = prop_df['regionidcity'].value_counts().to_dict()
    prop_df['N-city_count'] = prop_df['regionidcity'].map(city_count)
    
    prop_df['N-GarPoolAC'] = ((prop_df['garagecarcnt']>0) & (prop_df['pooltypeid10']>0) & (prop_df['airconditioningtypeid']!=5))*1
           
    # Mean square feet of neighborhood properties
    meanarea = prop_df.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()
    prop_df['mean_area'] = prop_df['regionidneighborhood'].map(meanarea)
    
    # Median year of construction of neighborhood properties
    medyear = prop_df.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()
    prop_df['med_year'] = prop_df['regionidneighborhood'].map(medyear)
    
    # Neighborhood latitude and longitude
    medlat = prop_df.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()
    prop_df['med_lat'] = prop_df['regionidneighborhood'].map(medlat)
    
    medlong = prop_df.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()
    prop_df['med_long'] = prop_df['regionidneighborhood'].map(medlong)
    
def remove_outliers(df):
    print('\nRemoving outliers ...')

    df.drop(df[df['logerror'] > 0.419].index, axis=0, inplace=True)
    df.drop(df[df['logerror'] <-0.4].index, axis=0, inplace=True)
    
    #df.drop(df[(df['finishedsquarefeet15']>20000) & (df['logerror']<1)].index, axis=0, inplace=True)
    #df.drop(df[(df['garagecarcnt']>20) & (df['logerror']<0)].index, axis=0, inplace=True)
    #df.drop(df[(df['garagetotalsqft']>6000) & (df['logerror']<0)].index, axis=0, inplace=True)
    #df.drop(df[(df['lotsizesquarefeet']>6000000) & (df['logerror']<2)].index, axis=0, inplace=True)
    #df.drop(df[(df['poolsizesum']>1500) & (df['logerror']<2)].index, axis=0, inplace=True)
    #df.drop(df[(df['unitcnt']>60) & (df['logerror']<2)].index, axis=0, inplace=True)
    #df.drop(df[(df['taxdelinquencyyear']>80) & (df['logerror']<2)].index, axis=0, inplace=True)
   
def select_features(df):
    print('\nSelecting important features ...')
    
    drop_features = ['parcelid', 'transactiondate',
                     'regionidcounty', 
                     'poolsizesum', 
                     'yardbuildingsqft26', 
                     'decktypeid', 
                     'storytypeid', 
                     'pooltypeid2',
                     ]
    
    df.drop(drop_features, axis=1, inplace=True)

def _process_data():
    print('\n\nSTEP1: processing data ...')
        
    global train_x
    global train_y
    global valid_x
    global valid_y
    global test_x
    
    valid_x = None
    valid_y = None
        
    # load data
    print('\nLoading data ...')
    train06_df = pd.read_csv(train06_path, parse_dates=["transactiondate"])
    train07_df = pd.read_csv(train07_path, parse_dates=["transactiondate"])
        
    prop06_df = pd.read_csv(properties06_path, low_memory = False)
    prop07_df = pd.read_csv(properties07_path, low_memory = False)
        
    # fill NA
    fill_NA(prop06_df)
    fill_NA(prop07_df)
    
    # encode features
    encode_features(prop06_df)
    encode_features(prop07_df)
   
    # add features
    add_features(prop06_df)
    add_features(prop07_df)
    
    # merge data
    train06_x = train06_df.merge(prop06_df, how='left', on='parcelid')
    train07_x = train07_df.merge(prop07_df, how='left', on='parcelid')
    train_x = pd.concat([train06_x, train07_x], axis=0)
    
    # analyze (optional)
    if False:
        analyze(train_x)
        
    # remove outliers
    remove_outliers(train_x)
        
    # select features
    select_features(train_x)
    
    # prepare train and valid data
    print('\nPreparing train and valid data ...')
    
    train_y = train_x['logerror']
    train_x.drop(['logerror'], axis=1, inplace=True)

    if USE_VALID_DATA is True:
        pass
        #select_qtr4 = pd.to_datetime(train06_df["transactiondate"]).dt.month>9
        #train06_x, valid06_x = train06_x[~select_qtr4], train06_x[select_qtr4]
        #train06_y, valid06_y = train06_y[~select_qtr4], train06_y[select_qtr4]
        
    print('train x shape: ', train_x.shape)
    print('train y shape: ', train_y.shape)
    if valid_x is not None:
        print('valid x shape: ', valid_x.shape)
        print('valid y shape: ', valid_y.shape)
    else:
        print('valid x shape: None')
        print('valid y shape: None')            
    
    # prepare test data
    print('\nPreparing test data ...')
    
    test_x = prop06_df[train_x.columns]
        
    print('test x shape: ', test_x.shape)
        
    # release
    del train06_df
    del prop06_df
    del train07_df
    del prop07_df    
    gc.collect()
    

## STEPZZZ: tune hyperparameters
def tune_eta(params):
    print('\nTuning eta ...')
    
    num_boost_round = 5000
    d_train = xgb.DMatrix(train_x, label=train_y)
    
    eta_list = [0.2, 0.1, 0.05, 0.025, 0.005, 0.0025]
    min_mae = float("Inf")
    best_eta = eta_list[0]
    
    for eta in eta_list:
        # update params
        params['eta'] = eta

        # run cv
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round = num_boost_round,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
    
        # print
        mae = cv_results['test-mae-mean'].min()
        rounds = cv_results['test-mae-mean'].argmin()
        print('eta:', eta, '; mae:',mae, '; rounds:', rounds)
    
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_eta = eta
                    
    print('best eta:', best_eta)
    print('min mae:', min_mae)
    return best_eta

def tune_max_depth__min_child_weight(params):
    print('\nTuning max_depth and min_child_weight ...')
    
    num_boost_round = 5000
    d_train = xgb.DMatrix(train_x, label=train_y)
    
    max_depth_list = list(range(5,10))
    min_child_weight_list = list(range(1,5))
    min_mae = float("Inf")
    best_max_depth = max_depth_list[0]
    best_min_child_weight = min_child_weight_list[0]
    
    for max_depth, min_child_weight in zip(max_depth_list, min_child_weight_list):
        # update params
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        
        # run cv
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round = num_boost_round,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        
        # print
        mae = cv_results['test-mae-mean'].min()
        rounds = cv_results['test-mae-mean'].argmin()
        print('max_depth:', max_depth, '; min_child_weight:', min_child_weight, '; mae:',mae, '; rounds:', rounds)
    
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_max_depth = max_depth
            best_min_child_weight = min_child_weight
                    
    print('best max_depth:', best_max_depth)
    print('best min_child_weight:', best_min_child_weight)
    print('min mae:', min_mae)
    return best_max_depth, best_min_child_weight

def tune_subsample__colsample_bytree(params):
    print('\nTuning subsample and colsample_bytree ...')
    
    num_boost_round = 5000
    d_train = xgb.DMatrix(train_x, label=train_y)
    
    subsample_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    colsample_bytree = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    min_mae = float("Inf")
    best_subsample = subsample_list[0]
    best_colsample_bytree = colsample_bytree[0]
    
    for subsample, colsample_bytree in zip(subsample_list, colsample_bytree):
        # update params
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        
        # run cv
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round = num_boost_round,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        
        # print
        mae = cv_results['test-mae-mean'].min()
        rounds = cv_results['test-mae-mean'].argmin()
        print('subsample:', subsample, '; colsample_bytree:', colsample_bytree, '; mae:',mae, '; rounds:', rounds)
    
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_subsample = subsample
            best_colsample_bytree = colsample_bytree
                    
    print('best subsample:', best_subsample)
    print('best colsample_bytree:', best_colsample_bytree)
    print('min mae:', min_mae)
    return best_subsample, best_colsample_bytree
        
def tune_alpha_lambda(params):
    print('\nTuning alpha and lambda ...')
    
    num_boost_round = 5000
    d_train = xgb.DMatrix(train_x, label=train_y)
    
    alpha_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lambda_list = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    min_mae = float("Inf")
    best_alpha = alpha_list[0]
    best_lambda = lambda_list[0]
    
    for alpha, lambdaa in zip(alpha_list, lambda_list):
        # update params
        params['alpha'] = alpha
        params['lambda'] = lambdaa
        
        # run cv
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round = num_boost_round,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        
        # print
        mae = cv_results['test-mae-mean'].min()
        rounds = cv_results['test-mae-mean'].argmin()
        print('alpha:', alpha, '; lambda:', lambdaa, '; mae:',mae, '; rounds:', rounds)
    
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_alpha = alpha
            best_lambda = lambdaa
                    
    print('best alpha:', best_alpha)
    print('best lambda:', best_lambda)
    print('min mae:', min_mae)
    return alpha, lambdaa
    
def _tune_params():
    params = {
        'eta': 0.01,
        #'max_depth': 7, 
        #'subsample': 0.6,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        #'lambda': 5.0,
        #'alpha': 0.65,
        #'colsample_bytree': 0.5,
        #'silent': 1
    }

    # tune eta
    if True:
        best_eta = tune_eta(params)
        params['eta'] = best_eta
    
    # tune max_depth and min_child_weight
    if True:
        best_max_depth, best_min_child_weight = tune_max_depth__min_child_weight(params)    
        params['max_depth'] = best_max_depth
        params['min_child_weight'] = best_min_child_weight
    
    # tune subsample and colsample_bytree
    if True:
        best_subsample, best_colsample_bytree = tune_subsample__colsample_bytree(params)
        params['subsample'] = best_subsample
        params['colsample_bytree'] = best_colsample_bytree
    
    # tune alpha and lambda
    if True:
        best_alpha, best_lambda = tune_alpha_lambda(params)
        params['subsample'] = best_alpha
        params['colsample_bytree'] = best_lambda

        
## STEP2: build model
def _build_model():
    print('\n\nSTEP2: building model ...')
    
    global best_num_boost_round
    best_num_boost_round = 1778
    
    # xgboost params
    global xgb_params
    xgb_params = {
        'eta': 0.1,
        'max_depth': 7, 
        'subsample': 0.6,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 5.0,
        'alpha': 0.65,
        'colsample_bytree': 0.5,
        'silent': 1
    }
    
    global full_xgb_params
    full_xgb_params = {
        'eta': 0.005,
        'max_depth': 6,
        'min_child_weight' : 2,
        'subsample': 0.4,
        'colsample_bytree': 0.4,
        'alpha': 0.8,
        'lambda': 8.0,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
    }
    
    # lightgbm params
    global lgb_params
    lgb_params = {
        'max_bin' : 10,
        'learning_rate' : 0.0021, # shrinkage_rate
        'boosting_type' : 'gbdt',
        'objective' : 'regression',
        'metric' : 'mae',
        'sub_feature' : 0.2,      # feature_fraction -- OK, back to .5, but maybe later increase this
        'bagging_fraction' : 0.85, # sub_row
        'bagging_freq' : 40, 
        'num_leaves' : 512,        # num_leaf
        'min_data' : 500,         # min_data_in_leaf
        'min_hessian' : 0.05,     # min_sum_hessian_in_leaf
        'verbose' : 0,
    }
    
    
## STEP3: train    
def _train():
    print('\n\nSTEP3: training ...')
    
    global xgb_clf
    global lgb_clf
    
    if USE_VALID_DATA is True:
        # xgboost
        d_train = xgb.DMatrix(train_x, label=train_y)
        d_valid = xgb.DMatrix(valid_x, label=valid_y)
        evals = [(d_train, 'train'), (d_valid, 'valid')]
        xgb_clf = xgb.train(xgb_params, d_train, 
                            num_boost_round=best_num_boost_round, evals=evals, 
                            early_stopping_rounds=100, verbose_eval=10)
        
        # ligtgbm
        d_train = lgb.Dataset(train_x, label=train_y)
        d_valid = lgb.Dataset(valid_x, label=valid_y)
        valid_sets = [d_train, d_valid]
        valid_names = ['train', 'valid']
        lgb_clf = lgb.train(lgb_params, d_train, 
                            num_boost_round=5000, 
                            valid_sets = valid_sets, valid_names = valid_names,
                            early_stopping_rounds=100,verbose_eval=10)
    else:
        # xgboost
        d_train = xgb.DMatrix(train_x, label=train_y)
        evals = [(d_train, 'train')]
        xgb_clf = xgb.train(full_xgb_params, d_train, 
                            num_boost_round=5000, 
                            evals=evals,verbose_eval=10)
    
            
## STEP4: predict
def _predict():
    print('\n\nSTEP4: predicting ...')
    
    global xgb_pred
    global lgb_pred
    
    if USE_VALID_DATA is True:
        # xgboost
        d_test = xgb.DMatrix(test_x)
        xgb_pred = xgb_clf.predict(d_test)
    
        # lightgbm
        test_x.values.astype(np.float32, copy=False)
        lgb_pred = lgb_clf.predict(test_x)
    else:
        # xgboost
        d_test = xgb.DMatrix(test_x)
        xgb_pred = xgb_clf.predict(d_test)
    
    
## STEP5: generate submission    
def _generate_submission():
    print('\n\nSTEP5: generating submission ...')

    submission = pd.read_csv(submission_path)
    for c in submission.columns[submission.columns != 'ParcelId']:
        if USE_VALID_DATA is True:
            submission[c] = xgb_pred*XGB_WEIGHT + lgb_pred*LGB_WEIGHT
        else:
            submission[c] = xgb_pred            
        
    submission.to_csv('sub{}.csv'.format(datetime.now().\
                strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.5f')


## main
def main():
    _process_data()
    if TUNE_PARAMS is True:
        _tune_params()
    else:
        _build_model()
        _train()
        _predict()
        _generate_submission()
    

if __name__ == "__main__":
    main()
    print('\n\n\nThe end.')
    