# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, skew
from sklearn import model_selection, preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime


import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

np.set_printoptions(threshold=np.nan)
color = sns.color_palette()


## STEP0: do setting
class Settings(Enum):
    global train_path
    global properties_path
    global submission_path
    global submission_dir
    
    train_path = 'C:/data/kaggle/zillow_price/train_2016_v2.csv'
    properties_path = 'C:/data/kaggle/zillow_price/properties_2016.csv'
    submission_path = "C:/data/kaggle/zillow_price/sample_submission.csv"
    submission_dir = "C:/data/kaggle/zillow_price/training_result/"
    
    def __str__(self):
        return self.value
        
## STEP1: process data    
# helper: display transaction data    
def display_transaction_date():
    plt.figure()
    counts_in_month = train_df['transactiondate'].dt.month.value_counts()
    sns.barplot(x=counts_in_month.index, y=counts_in_month.values, alpha=0.8, color=color[3])
    plt.xticks(rotation='vertical')
    plt.xlabel('Month of transaction', fontsize=12)
    plt.ylabel('Number of transactions', fontsize=12)
    plt.show()

# helper: display distribution
def display_distrib(feature):
    plt.figure()
    sns.distplot(train_df[feature].dropna() , fit=norm);
    (mu, sigma) = norm.fit(train_df[feature].dropna())    
    
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('distribution')
    plt.show()
    
# helper: display outlier respect to the target feature
def display_features_with_target(feature):
    fig, ax = plt.subplots()
    ax.scatter(x = train_df[feature], y = train_df['logerror'])
    plt.ylabel('logerror', fontsize=13)
    plt.xlabel(feature, fontsize=13)
    plt.show()  
    
# normalize the distribution of the target feature if necessary    
def process_target_feature():
    plt.figure()
    sns.distplot(train_df['logerror'] , fit=norm);
    (mu, sigma) = norm.fit(train_df['logerror'])
    
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('distribution')
    plt.show()

# fill NAs
def process_NA():
    # compute NA ratio before adding
    #na_ratio = (train_df.isnull().sum() / len(train_df)) * 100    
    #na_ratio = na_ratio.sort_values(ascending=False)
    #print('\nNA ratio before adding:\n', na_ratio)
    
    # add NA manually
    train_df['airconditioningtypeid'] = train_df['airconditioningtypeid'].fillna("None")
    train_df['architecturalstyletypeid'] = train_df['architecturalstyletypeid'].fillna("None")
    train_df['basementsqft'] = train_df['basementsqft'].fillna(0)
    train_df['bathroomcnt'] = train_df['bathroomcnt'].fillna(train_df['bathroomcnt'].mean(axis=0))
    train_df['buildingclasstypeid'] = train_df['buildingclasstypeid'].fillna("None")
    train_df['buildingqualitytypeid'] = train_df['buildingqualitytypeid'].fillna("None")
    train_df['calculatedbathnbr'] = train_df['calculatedbathnbr'].fillna(0)
    train_df['decktypeid'] = train_df['decktypeid'].fillna("None")
    train_df['finishedfloor1squarefeet'] = train_df['finishedfloor1squarefeet'].fillna(0)
    train_df['calculatedfinishedsquarefeet'] = train_df['calculatedfinishedsquarefeet'].fillna(0)
    train_df['finishedsquarefeet12'] = train_df['finishedsquarefeet12'].fillna(0)
    train_df['finishedsquarefeet13'] = train_df['finishedsquarefeet13'].fillna(0)
    train_df['finishedsquarefeet15'] = train_df['finishedsquarefeet15'].fillna(0)
    train_df['finishedsquarefeet50'] = train_df['finishedsquarefeet50'].fillna(0)
    train_df['finishedsquarefeet6'] = train_df['finishedsquarefeet6'].fillna(0)
    train_df['fips'] = train_df['fips'].fillna(0)
    train_df['fireplacecnt'] = train_df['fireplacecnt'].fillna(0)
    train_df['fullbathcnt'] = train_df['fullbathcnt'].fillna(0)
    train_df['garagecarcnt'] = train_df['garagecarcnt'].fillna(0)
    train_df['garagetotalsqft'] = train_df['garagetotalsqft'].fillna(0)
    train_df['hashottuborspa'] = train_df['hashottuborspa'].fillna("FALSE")
    train_df['heatingorsystemtypeid'] = train_df['heatingorsystemtypeid'].fillna("None")
    train_df['latitude'] = train_df['latitude'].fillna(train_df['latitude'].mean(axis=0))
    train_df['longitude'] = train_df['longitude'].fillna(train_df['longitude'].mean(axis=0))
    train_df['lotsizesquarefeet'] = train_df['lotsizesquarefeet'].fillna(0)
    train_df['poolcnt'] = train_df['poolcnt'].fillna(0)
    train_df['poolsizesum'] = train_df['poolsizesum'].fillna(0)
    train_df['pooltypeid10'] = train_df['pooltypeid10'].fillna("None")
    train_df['pooltypeid2'] = train_df['pooltypeid2'].fillna("None")
    train_df['pooltypeid7'] = train_df['pooltypeid7'].fillna("None")
    train_df['propertycountylandusecode'] = train_df['propertycountylandusecode'].fillna("None")
    train_df['propertylandusetypeid'] = train_df['propertylandusetypeid'].fillna("None")
    train_df['propertyzoningdesc'] = train_df['propertyzoningdesc'].fillna("None")
    train_df['rawcensustractandblock'] = train_df['rawcensustractandblock'].fillna(train_df['rawcensustractandblock'].mean(axis=0))
    train_df['regionidcity'] = train_df['regionidcity'].fillna("None")
    train_df['regionidcounty'] = train_df['regionidcounty'].fillna("None")
    train_df['regionidneighborhood'] = train_df['regionidneighborhood'].fillna("None")
    train_df['regionidzip'] = train_df['regionidzip'].fillna("None")
    train_df['roomcnt'] = train_df['roomcnt'].fillna(0)
    train_df['storytypeid'] = train_df['storytypeid'].fillna("None")
    train_df['threequarterbathnbr'] = train_df['threequarterbathnbr'].fillna(0)
    train_df['typeconstructiontypeid'] = train_df['typeconstructiontypeid'].fillna("None")
    train_df['unitcnt'] = train_df['unitcnt'].fillna(0)
    train_df['yardbuildingsqft17'] = train_df['yardbuildingsqft17'].fillna(0)
    train_df['yardbuildingsqft26'] = train_df['yardbuildingsqft26'].fillna(0)
    train_df['yearbuilt'] = train_df['yearbuilt'].fillna("None")
    train_df['numberofstories'] = train_df['numberofstories'].fillna(0)
    train_df['fireplaceflag'] = train_df['fireplaceflag'].fillna("FALSE")
    train_df['structuretaxvaluedollarcnt'] = train_df['structuretaxvaluedollarcnt'].fillna(train_df['structuretaxvaluedollarcnt'].mean(axis=0))
    train_df['taxvaluedollarcnt'] = train_df['taxvaluedollarcnt'].fillna(train_df['taxvaluedollarcnt'].mean(axis=0))
    train_df['assessmentyear'] = train_df['assessmentyear'].fillna("None")
    train_df['landtaxvaluedollarcnt'] = train_df['landtaxvaluedollarcnt'].fillna(train_df['landtaxvaluedollarcnt'].mean(axis=0))
    train_df['taxamount'] = train_df['taxamount'].fillna(train_df['taxamount'].mean(axis=0))
    train_df['taxdelinquencyflag'] = train_df['taxdelinquencyflag'].fillna("N")
    train_df['taxdelinquencyyear'] = train_df['taxdelinquencyyear'].fillna("None")
    train_df['censustractandblock'] = train_df['censustractandblock'].fillna(train_df['censustractandblock'].mean(axis=0))
    
    prop_df['airconditioningtypeid'] = prop_df['airconditioningtypeid'].fillna("None")
    prop_df['architecturalstyletypeid'] = prop_df['architecturalstyletypeid'].fillna("None")
    prop_df['basementsqft'] = prop_df['basementsqft'].fillna(0)
    prop_df['bathroomcnt'] = prop_df['bathroomcnt'].fillna(prop_df['bathroomcnt'].mean(axis=0))
    prop_df['buildingclasstypeid'] = prop_df['buildingclasstypeid'].fillna("None")
    prop_df['buildingqualitytypeid'] = prop_df['buildingqualitytypeid'].fillna("None")
    prop_df['calculatedbathnbr'] = prop_df['calculatedbathnbr'].fillna(0)
    prop_df['decktypeid'] = prop_df['decktypeid'].fillna("None")
    prop_df['finishedfloor1squarefeet'] = prop_df['finishedfloor1squarefeet'].fillna(0)
    prop_df['calculatedfinishedsquarefeet'] = prop_df['calculatedfinishedsquarefeet'].fillna(0)
    prop_df['finishedsquarefeet12'] = prop_df['finishedsquarefeet12'].fillna(0)
    prop_df['finishedsquarefeet13'] = prop_df['finishedsquarefeet13'].fillna(0)
    prop_df['finishedsquarefeet15'] = prop_df['finishedsquarefeet15'].fillna(0)
    prop_df['finishedsquarefeet50'] = prop_df['finishedsquarefeet50'].fillna(0)
    prop_df['finishedsquarefeet6'] = prop_df['finishedsquarefeet6'].fillna(0)
    prop_df['fips'] = prop_df['fips'].fillna(0)
    prop_df['fireplacecnt'] = prop_df['fireplacecnt'].fillna(0)
    prop_df['fullbathcnt'] = prop_df['fullbathcnt'].fillna(0)
    prop_df['garagecarcnt'] = prop_df['garagecarcnt'].fillna(0)
    prop_df['garagetotalsqft'] = prop_df['garagetotalsqft'].fillna(0)
    prop_df['hashottuborspa'] = prop_df['hashottuborspa'].fillna("FALSE")
    prop_df['heatingorsystemtypeid'] = prop_df['heatingorsystemtypeid'].fillna("None")
    prop_df['latitude'] = prop_df['latitude'].fillna(prop_df['latitude'].mean(axis=0))
    prop_df['longitude'] = prop_df['longitude'].fillna(prop_df['longitude'].mean(axis=0))
    prop_df['lotsizesquarefeet'] = prop_df['lotsizesquarefeet'].fillna(0)
    prop_df['poolcnt'] = prop_df['poolcnt'].fillna(0)
    prop_df['poolsizesum'] = prop_df['poolsizesum'].fillna(0)
    prop_df['pooltypeid10'] = prop_df['pooltypeid10'].fillna("None")
    prop_df['pooltypeid2'] = prop_df['pooltypeid2'].fillna("None")
    prop_df['pooltypeid7'] = prop_df['pooltypeid7'].fillna("None")
    prop_df['propertycountylandusecode'] = prop_df['propertycountylandusecode'].fillna("None")
    prop_df['propertylandusetypeid'] = prop_df['propertylandusetypeid'].fillna("None")
    prop_df['propertyzoningdesc'] = prop_df['propertyzoningdesc'].fillna("None")
    prop_df['rawcensustractandblock'] = prop_df['rawcensustractandblock'].fillna(prop_df['rawcensustractandblock'].mean(axis=0))
    prop_df['regionidcity'] = prop_df['regionidcity'].fillna("None")
    prop_df['regionidcounty'] = prop_df['regionidcounty'].fillna("None")
    prop_df['regionidneighborhood'] = prop_df['regionidneighborhood'].fillna("None")
    prop_df['regionidzip'] = prop_df['regionidzip'].fillna("None")
    prop_df['roomcnt'] = prop_df['roomcnt'].fillna(0)
    prop_df['storytypeid'] = prop_df['storytypeid'].fillna("None")
    prop_df['threequarterbathnbr'] = prop_df['threequarterbathnbr'].fillna(0)
    prop_df['typeconstructiontypeid'] = prop_df['typeconstructiontypeid'].fillna("None")
    prop_df['unitcnt'] = prop_df['unitcnt'].fillna(0)
    prop_df['yardbuildingsqft17'] = prop_df['yardbuildingsqft17'].fillna(0)
    prop_df['yardbuildingsqft26'] = prop_df['yardbuildingsqft26'].fillna(0)
    prop_df['yearbuilt'] = prop_df['yearbuilt'].fillna("None")
    prop_df['numberofstories'] = prop_df['numberofstories'].fillna(0)
    prop_df['fireplaceflag'] = prop_df['fireplaceflag'].fillna("FALSE")
    prop_df['structuretaxvaluedollarcnt'] = prop_df['structuretaxvaluedollarcnt'].fillna(prop_df['structuretaxvaluedollarcnt'].mean(axis=0))
    prop_df['taxvaluedollarcnt'] = prop_df['taxvaluedollarcnt'].fillna(prop_df['taxvaluedollarcnt'].mean(axis=0))
    prop_df['assessmentyear'] = prop_df['assessmentyear'].fillna("None")
    prop_df['landtaxvaluedollarcnt'] = prop_df['landtaxvaluedollarcnt'].fillna(prop_df['landtaxvaluedollarcnt'].mean(axis=0))
    prop_df['taxamount'] = prop_df['taxamount'].fillna(prop_df['taxamount'].mean(axis=0))
    prop_df['taxdelinquencyflag'] = prop_df['taxdelinquencyflag'].fillna("N")
    prop_df['taxdelinquencyyear'] = prop_df['taxdelinquencyyear'].fillna("None")
    prop_df['censustractandblock'] = prop_df['censustractandblock'].fillna(prop_df['censustractandblock'].mean(axis=0))
    
    # compute NA ratio after adding
    #na_ratio = (train_df.isnull().sum() / len(train_df)) * 100    
    #na_ratio = na_ratio.sort_values(ascending=False)
    #print('\nNA ratio after adding:\n', na_ratio)
        
# remove outliers
def process_outlier():
    # display before removing outliers
    #for feature in train_df:
    #    if train_df[feature].dtype == "float64":
    #        display_features_with_target(feature)          
    
    #display_features_with_target('finishedfloor1squarefeet')
    train_df.drop(train_df[(train_df['finishedfloor1squarefeet']<2000) & (train_df['logerror']<-4)].index, axis=0, inplace=True)
    #display_features_with_target('finishedfloor1squarefeet')
    
    #display_features_with_target('finishedsquarefeet15')
    train_df.drop(train_df[(train_df['finishedsquarefeet15']>20000) & (train_df['logerror']<1)].index, axis=0, inplace=True)
    #display_features_with_target('finishedsquarefeet15')
    
    #display_features_with_target('finishedsquarefeet50')
    train_df.drop(train_df[(train_df['finishedsquarefeet50']<2000) & (train_df['logerror']<-4)].index, axis=0, inplace=True)
    #display_features_with_target('finishedsquarefeet50')
    
    #display_features_with_target('garagetotalsqft')
    train_df.drop(train_df[(train_df['garagetotalsqft']>7000) & (train_df['logerror']>-1)].index, axis=0, inplace=True)
    #display_features_with_target('garagetotalsqft')    
    
    #display_features_with_target('lotsizesquarefeet')
    train_df.drop(train_df[(train_df['lotsizesquarefeet']>6000000) & (train_df['logerror']>-1)].index, axis=0, inplace=True)
    #display_features_with_target('lotsizesquarefeet')   
    
    #display_features_with_target('unitcnt')
    train_df.drop(train_df[(train_df['unitcnt']>40) & (train_df['logerror']<2)].index, axis=0, inplace=True)
    #display_features_with_target('unitcnt')     

    #display_features_with_target('yardbuildingsqft17')
    train_df.drop(train_df[(train_df['yardbuildingsqft17']<500) & (train_df['logerror']<-4)].index, axis=0, inplace=True)
    #display_features_with_target('yardbuildingsqft17')   
    
    #display_features_with_target('taxdelinquencyyear')
    train_df.drop(train_df[(train_df['taxdelinquencyyear']>80) & (train_df['logerror']>0)].index, axis=0, inplace=True)
    #display_features_with_target('taxdelinquencyyear')   
    
# normalize distribution
def process_skewness():                
    #display_distrib("calculatedfinishedsquarefeet")            
    train_df["calculatedfinishedsquarefeet"] = np.log1p(train_df["calculatedfinishedsquarefeet"])
    #display_distrib("calculatedfinishedsquarefeet")
            
    #display_distrib("finishedsquarefeet12")            
    train_df["finishedsquarefeet12"] = np.log1p(train_df["finishedsquarefeet12"])
    #display_distrib("finishedsquarefeet12")
            
# do feature engineering: add, select, encode
def process_feature():
    # remove features
    transactiondate = train_df['transactiondate']
    train_df.drop(['transactiondate'], axis=1, inplace=True)
    
    train_df.drop(['parcelid'], axis=1, inplace=True)    
    prop_df.drop(['parcelid'], axis=1, inplace=True)
    
    # encode features: transform categorical features into numeric features
    for feature in train_df:
        if train_df[feature].dtype=='object':
            lbl = preprocessing.LabelEncoder()            
            lbl.fit(list(train_df[feature].values)) 
            train_df[feature] = lbl.transform(list(train_df[feature].values))
            
    for feature in prop_df:
        if prop_df[feature].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(prop_df[feature].values)) 
            prop_df[feature] = lbl.transform(list(prop_df[feature].values))
            
    # run xgboot to pick best features up
    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    
    train_y = train_df.logerror.values
    dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
    feature_importance = model.get_fscore()

    features_df = pd.DataFrame()
    features_df['features'] = feature_importance.keys()
    features_df['importance'] = feature_importance.values()
    features_df.sort_values(by=['importance'], ascending=False, inplace=True)
    
    '''
    # display best features
    fig,ax= plt.subplots()
    fig.set_size_inches(10,10)
    plt.xticks(rotation=90)
    sns.barplot(data=features_df, x="importance", y="features", ax=ax, orient="h", color="#34495e")
    plt.show()
    '''
	
    # select only important features
    for feature in train_df:
        if feature not in list(features_df.head(40).features):
            train_df.drop(feature, axis=1, inplace=True)
            prop_df.drop(feature, axis=1, inplace=True)
        
    # add features
    train_df['transactiondate_year'] = transactiondate.dt.year
    train_df['transactiondate_month'] = transactiondate.dt.month
    train_df['transactiondate_year'].fillna(0)
    train_df['transactiondate_month'].fillna(0)        
        
def _process_data():
    print('\n\nSTEP1: _process_data() ...')
    
    # load data
    print('\nload data ...')
    global train_df
    global prop_df
    train_df = pd.read_csv(train_path, parse_dates=["transactiondate"])
    prop_df = pd.read_csv(properties_path)

    #print(pd_train.head(5))
    #print(pd_prop.head(5))
    
    print('train shape before merging: ', train_df.shape)
    print('prop shape before merging: ', prop_df.shape)
    train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
    print('train shape after merging: ', train_df.shape)
    print('prop shape after merging: ', prop_df.shape)
    
    # process the target feature (logerror)
    #process_target_feature()
    #print('train shape after procesing the target feature: ', train_df.shape)
    
    # process transactiondate
    #display_transaction_date()
    
    # remove outliers
    print('\nprocess outliers ...')
    process_outlier()
    print('train shape after processing outlier: ', train_df.shape)
    print('prop shape after processing outlier: ', prop_df.shape)
    
    # fill NAs
    print('\nprocess NA ...')
    process_NA()
    print('train shape after processing NA: ', train_df.shape)
    print('prop shape after processing NA: ', prop_df.shape)
    
    # normalize distribution
    print('\nprocess skewness ...')
    process_skewness()
    print('train shape after processing skewness: ', train_df.shape)
    print('prop shape after processing skewness: ', prop_df.shape)
    
    # do feature engineering: add, select, encode
    print('\nprocess feature ...')
    process_feature()
    print('train shape after processing feature: ', train_df.shape)
    print('prop shape after processing feature: ', prop_df.shape)
    
    # prepare data
    print('\nprepare data ...')
    logerror = train_df['logerror']
    train_df.drop(['logerror'], axis=1, inplace=True)
    
    global train_x
    global train_y
    global test_x
    train_y = logerror.values
    train_x = train_df
    test_x = prop_df
    
    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    print('test_x shape: ', test_x.shape)
    
    #print('train features: ', train_x.columns.values)
    #print('test features: ', test_x.columns.values)
    
## STEP2: build model
# compute root mean square for cross evaluation
def rmsle_cv(model):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)
    rmse= np.sqrt(-cross_val_score(model, train_df.values, train_y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# build model
def _build_model():
    print('\n\nSTEP2: _build_model() ...')
    
    # lightgbm model
    global lgb_model
    lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin = 55, bagging_fraction = 0.8,
                                  bagging_freq = 5, feature_fraction = 0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
        
    score = rmsle_cv(lgb_model)
    print('LGBM score(cv): {:.4f} ({:.4f})'.format(score.mean(), score.std()))
    
    # xgboost model
    global xgb_model
    xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                                 learning_rate=0.05, max_depth=3, 
                                 min_child_weight=1.7817, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1)
    
    score = rmsle_cv(xgb_model)
    print('Xgboost score(cv): {:.4f} ({:.4f})'.format(score.mean(), score.std()))  

## STEP3: train
# compute mean absolute error
def MAE(y, y_pred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-y_pred[i]) for i in range(len(y))]) / len(y)

# train
def _train():
    print('\n\nSTEP3: _train() ...')
    
    # lightgbm training
    lgb_model.fit(train_x, train_y)
    lgb_train_pred = lgb_model.predict(train_x)    
    print('lightgbm MAE: {:.4f}'.format(MAE(train_y, lgb_train_pred)))
    
    # xgboost training
    xgb_model.fit(train_x, train_y)
    xgb_train_pred = xgb_model.predict(train_x)    
    print('xgboost MAE: {:.4f}'.format(MAE(train_y, xgb_train_pred)))    

## predict
def _predict():
    print('\n\nSTEP4: _predict() ...')
    
    global submission
    
    submission = pd.read_csv(submission_path)
    test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
    test_columns = ['201610','201611','201612','201710','201711','201712']
    
    for i in range(len(test_dates)):
        test_x["transactiondate_year"] = pd.to_datetime(test_dates[i]).year
        test_x["transactiondate_month"] = pd.to_datetime(test_dates[i]).month
        test_x['transactiondate_year'].fillna(0)
        test_x['transactiondate_month'].fillna(0)
        
        #lgb_test_pred = lgb_model.predict(test_x)
        #submission[test_columns[i]] = [float(format(x, '.4f')) for x in lgb_test_pred]
        
        xgb_test_pred = xgb_model.predict(test_x)
        submission[test_columns[i]] = [float(format(x, '.4f')) for x in xgb_test_pred]
        
## STEP5: generate submission
def _generate_submission():
    print('\n\nSTEP5: _generate_submission() ...')
    submission.to_csv(submission_dir+'sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
        
    
## main
def main():
    _process_data()
    _build_model()
    _train()
    _predict()
    _generate_submission()
    

if __name__ == "__main__":
    main()
    print('\n\n\nThe end.')
    