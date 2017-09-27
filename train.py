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
import gc



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
    train_df.drop(['transactiondate'], axis=1, inplace=True)
    
    train_df.drop(['parcelid'], axis=1, inplace=True)    
    prop_df.drop(['parcelid'], axis=1, inplace=True)
    
    #TODO
    # encode features: transform categorical features into numeric features
    for feature in prop_df:
        if prop_df[feature].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(prop_df[feature].values)) 
            prop_df[feature] = lbl.transform(list(prop_df[feature].values))
            
    for feature in train_df:
        if train_df[feature].dtype=='object':
            lbl = preprocessing.LabelEncoder()            
            lbl.fit(list(train_df[feature].values)) 
            train_df[feature] = lbl.transform(list(train_df[feature].values))

                  
    '''
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
    
    # display best features
    fig,ax= plt.subplots()
    fig.set_size_inches(10,10)
    plt.xticks(rotation=90)
    sns.barplot(data=features_df, x="importance", y="features", ax=ax, orient="h", color="#34495e")
    plt.show()
	
    # select only important features
    for feature in train_df:
        if feature not in list(features_df.head(45).features):
            train_df.drop(feature, axis=1, inplace=True)
            prop_df.drop(feature, axis=1, inplace=True)
    '''
            
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
    global valid_x
    global valid_y
    global test_x
    
    train_y = logerror.values
    train_x = train_df
    test_x = prop_df

    split = 80000
    train_x, valid_x = train_x[:split], train_x[split:]
    train_y, valid_y = train_y[:split], train_y[split:]
    
    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    print('valid_x shape: ', valid_x.shape)
    print('valid_y shape: ', valid_y.shape)    
    print('test_x shape: ', test_x.shape)
    
    #print('train features: ', train_x.columns.values)
    #print('test features: ', test_x.columns.values)

    # release
    del train_df; gc.collect()
    del prop_df; gc.collect()
    

## STEP2: build model
def _build_model():
    print('\nBuild model ...')
    global params
    params = {}
    params['eta'] = 0.02
    params['objective'] = 'reg:linear'
    params['eval_metric'] = 'mae'
    params['max_depth'] = 4
    params['silent'] = 1    

    
## STEP3: train    
def _train():
    print('Train ...')
    global clf
    d_train = xgb.DMatrix(train_x, label=train_y)
    d_valid = xgb.DMatrix(valid_x, label=valid_y)
    
    evals = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 
                    num_boost_round=10000, evals=evals, 
                    early_stopping_rounds=100, verbose_eval=10)
    
    
## STEP4: predict    
def _predict():
    print('Predicting ...')
    global submission
    d_test = xgb.DMatrix(test_x)
    p_test = clf.predict(d_test)
    submission = pd.read_csv('C:/data/kaggle/zillow_price/sample_submission.csv')
    for c in submission.columns[submission.columns != 'ParcelId']:
        submission[c] = p_test
    
    
## STEP5: generate submission    
def _generate_submission():
    print('Writing csv ...')
    submission.to_csv(submission_dir+'sub{}.csv'.format(datetime.now().\
                      strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')


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
    