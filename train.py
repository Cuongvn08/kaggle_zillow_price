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
    global train_path
    global properties_path
    global submission_path
    global XGB_WEIGHT
    global LGB_WEIGHT
    
    train_path      = 'C:/data/kaggle/zillow_price/train_2016_v2.csv'
    properties_path = 'C:/data/kaggle/zillow_price/properties_2016.csv'
    submission_path = 'C:/data/kaggle/zillow_price/sample_submission.csv'
    XGB_WEIGHT = 0.5
    LGB_WEIGHT = 1 - XGB_WEIGHT
    
    
    def __str__(self):
        return self.value
        
    
## STEP1: process data
def analyze(df):
    print('\nAnalyzing ...')
    
    # show logerror
    if(0):
        plt.figure(figsize=(8,6))
        plt.scatter(range(df.shape[0]), np.sort(df.logerror.values))
        plt.xlabel('index', fontsize=12)
        plt.ylabel('logerror', fontsize=12)
        plt.show()
        
    # show other features in correlation with logerror
    if(0):
        for feature in df:
            if df[feature].dtype == "float64" or "int64":
                if feature != 'transactiondate':
                    fig, ax = plt.subplots()
                    ax.scatter(x = df[feature], y = df['logerror'])
                    plt.ylabel('logerror', fontsize=13)
                    plt.xlabel(feature, fontsize=13)
                    plt.show()  
                    
    # show correlation
    if(0):
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

    #prop_df['zip_std'] = prop_df['regionidzip'].map(zipstd)
    #prop_df['city_std'] = prop_df['regionidcity'].map(citystd)
    #prop_df['hood_std'] = prop_df['regionidneighborhood'].map(hoodstd)
    
def remove_outliers(df):        
    print('\nRemoving outliers ...')

    df.drop(df[df['logerror'] > 0.419].index, axis=0, inplace=True)
    df.drop(df[df['logerror'] <-0.4].index, axis=0, inplace=True)
    
    df.drop(df[(df['finishedsquarefeet15']>20000) & (df['logerror']<1)].index, axis=0, inplace=True)
    df.drop(df[(df['garagecarcnt']>20) & (df['logerror']<0)].index, axis=0, inplace=True)
    df.drop(df[(df['garagetotalsqft']>6000) & (df['logerror']<0)].index, axis=0, inplace=True)
    df.drop(df[(df['lotsizesquarefeet']>6000000) & (df['logerror']<2)].index, axis=0, inplace=True)
    df.drop(df[(df['poolsizesum']>1500) & (df['logerror']<2)].index, axis=0, inplace=True)
    df.drop(df[(df['unitcnt']>60) & (df['logerror']<2)].index, axis=0, inplace=True)
    df.drop(df[(df['taxdelinquencyyear']>80) & (df['logerror']<2)].index, axis=0, inplace=True)
           
def select_features(df):
    print('\nSelecting important features ...')
    
    #drop_features = ['parcelid', 'transactiondate',
    #                 'airconditioningtypeid', 'buildingclasstypeid',
    #                 'buildingqualitytypeid', 'regionidcity']
    #drop_features = ['parcelid', 'transactiondate',
    #                 'poolsizesum', 'storytypeid', 'typeconstructiontypeid',
    #                 'decktypeid', 'pooltypeid10',
    #                 ]
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
        
    # load data
    print('\nLoading data ...')
    train_df = pd.read_csv(train_path, parse_dates=["transactiondate"])
    prop_df = pd.read_csv(properties_path)
    
    # fill NA
    fill_NA(prop_df)
    
    # encode features
    encode_features(prop_df)
    
    # add features
    add_features(prop_df)
    
    # merge data
    train_x = train_df.merge(prop_df, how='left', on='parcelid')
    
    # analyze (optional)
    analyze(train_x)
    
    # remove outliers
    remove_outliers(train_x)
    
    # select features
    select_features(train_x)

    # prepare train and valid data
    print('\nPreparing train and valid data ...')
    
    train_y = train_x['logerror']
    train_x.drop(['logerror'], axis=1, inplace=True)
    
    select_qtr4 = pd.to_datetime(train_df["transactiondate"]).dt.month > 9
    train_x, valid_x = train_x[~select_qtr4], train_x[select_qtr4]
    train_y, valid_y = train_y[~select_qtr4], train_y[select_qtr4]
    
    print('train x shape: ', train_x.shape)
    print('train y shape: ', train_y.shape)
    print('valid x shape: ', valid_x.shape)
    print('valid y shape: ', valid_y.shape)
    
    # prepare test data
    print('\nPreparing test data ...')
    
    test_x = prop_df[train_x.columns]
    print('test x shape: ', test_x.shape)
    
    # release
    del train_df
    del prop_df
    gc.collect()
    
    
## STEP2: build model
def _build_model():
    print('\n\nSTEP2: building model ...')
    
    # xgboost params
    global xgb_params
    xgb_params = {
        'eta': 0.007,
        'max_depth': 7, 
        'subsample': 0.6,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 5.0,
        'alpha': 0.65,
        'colsample_bytree': 0.5,
        'silent': 1
    }
    
    # lightgbm params
    global lgb_params
    lgb_params = {
        'max_bin' : 10,
        'learning_rate' : 0.0021, # shrinkage_rate
        'boosting_type' : 'gbdt',
        'objective' : 'regression',
        'metric' : 'mae',
        'sub_feature' : 0.5,      # feature_fraction -- OK, back to .5, but maybe later increase this
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
    
    # xgboost
    global xgb_clf
    d_train = xgb.DMatrix(train_x, label=train_y)
    d_valid = xgb.DMatrix(valid_x, label=valid_y)
    evals = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_clf = xgb.train(xgb_params, d_train, num_boost_round=10000, evals=evals, 
                        early_stopping_rounds=100, verbose_eval=10)
    
    # ligtgbm
    global lgb_clf
    d_train = lgb.Dataset(train_x, label=train_y)
    d_valid = lgb.Dataset(valid_x, label=valid_y)
    valid_sets = [d_train, d_valid]
    valid_names = ['train', 'valid']
    lgb_clf = lgb.train(lgb_params, d_train, num_boost_round=10000, 
                       valid_sets = valid_sets, valid_names = valid_names,
                       early_stopping_rounds=100,verbose_eval=10)
        
    
## STEP4: predict
def _predict():
    print('\n\nSTEP4: predicting ...')
    
    global xgb_pred
    d_test = xgb.DMatrix(test_x)
    xgb_pred = xgb_clf.predict(d_test)
    
    global lgb_pred
    test_x.values.astype(np.float32, copy=False)
    lgb_pred = lgb_clf.predict(test_x)
    
    
## STEP5: generate submission    
def _generate_submission():
    print('\n\nSTEP5: generating submission ...')

    submission = pd.read_csv(submission_path)
    for c in submission.columns[submission.columns != 'ParcelId']:
        submission[c] = lgb_pred
        
    submission.to_csv('sub{}.csv'.format(datetime.now().\
                strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.5f')


## main
def main():
    _process_data()
    _build_model()
    _train()
    _predict()
    #_generate_submission()
    

if __name__ == "__main__":
    main()
    print('\n\n\nThe end.')
    