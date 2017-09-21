# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, skew
from sklearn import model_selection, preprocessing
import xgboost as xgb


import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

np.set_printoptions(threshold=np.nan)
color = sns.color_palette()


## STEP1: do setting
class Settings(Enum):
    global train_path
    global properties_path
    global zillow_data_dictionary_path
    
    train_path = 'C:/data/kaggle/zillow_price/train_2016_v2.csv'
    properties_path = 'C:/data/kaggle/zillow_price/properties_2016.csv'
	
    def __str__(self):
        return self.value
        
    
## STEP2: process data    
# helper: display transaction data    
def display_transaction_data():
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
def display_outlier(feature):
    fig, ax = plt.subplots()
    ax.scatter(x = train_df[feature], y = train_df['logerror'])
    plt.ylabel('logerror', fontsize=13)
    plt.xlabel(feature, fontsize=13)
    plt.show()  
    
# normalize the distribution of the target feature if necessary    
def process_target_feature():
    plt.figure()
    sns.distplot(train_df['logerror'].dropna() , fit=norm);
    (mu, sigma) = norm.fit(train_df['logerror'].dropna())    
    
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('distribution')
    plt.show()

# fill NAs
def process_NA():
    # compute NA ratio before adding
    na_ratio = (train_df.isnull().sum() / len(train_df)) * 100    
    na_ratio = na_ratio.sort_values(ascending=False)
    print('\nNA ratio before adding:\n', na_ratio)
    
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
    
    # compute NA ratio after adding
    na_ratio = (train_df.isnull().sum() / len(train_df)) * 100    
    na_ratio = na_ratio.sort_values(ascending=False)
    print('\nNA ratio after adding:\n', na_ratio)
        
# remove outliers
def process_outlier():
    print(train_df.dtypes)
    
    # display before removing outliers
    #for feature in train_df:
    #    if train_df[feature].dtype == "float64":
    #        display_outlier(feature)          
    
    display_outlier('finishedfloor1squarefeet')
    train_df.drop(train_df[(train_df['finishedfloor1squarefeet']<2000) & (train_df['logerror']<-4)].index, axis=0, inplace=True)
    display_outlier('finishedfloor1squarefeet')
    
    display_outlier('finishedsquarefeet15')
    train_df.drop(train_df[(train_df['finishedsquarefeet15']>20000) & (train_df['logerror']<1)].index, axis=0, inplace=True)
    display_outlier('finishedsquarefeet15')
    
    display_outlier('finishedsquarefeet50')
    train_df.drop(train_df[(train_df['finishedsquarefeet50']<2000) & (train_df['logerror']<-4)].index, axis=0, inplace=True)
    display_outlier('finishedsquarefeet50')
    
    display_outlier('garagetotalsqft')
    train_df.drop(train_df[(train_df['garagetotalsqft']>7000) & (train_df['logerror']>-1)].index, axis=0, inplace=True)
    display_outlier('garagetotalsqft')    
    
    display_outlier('lotsizesquarefeet')
    train_df.drop(train_df[(train_df['lotsizesquarefeet']>6000000) & (train_df['logerror']>-1)].index, axis=0, inplace=True)
    display_outlier('lotsizesquarefeet')   
    
    display_outlier('unitcnt')
    train_df.drop(train_df[(train_df['unitcnt']>40) & (train_df['logerror']<2)].index, axis=0, inplace=True)
    display_outlier('unitcnt')     

    display_outlier('yardbuildingsqft17')
    train_df.drop(train_df[(train_df['yardbuildingsqft17']<500) & (train_df['logerror']<-4)].index, axis=0, inplace=True)
    display_outlier('yardbuildingsqft17')   
    
    display_outlier('taxdelinquencyyear')
    train_df.drop(train_df[(train_df['taxdelinquencyyear']>80) & (train_df['logerror']>0)].index, axis=0, inplace=True)
    display_outlier('taxdelinquencyyear')   
    
# normalize distribution
def process_skewness():                
    display_distrib("calculatedfinishedsquarefeet")            
    train_df["calculatedfinishedsquarefeet"] = np.log1p(train_df["calculatedfinishedsquarefeet"])
    display_distrib("calculatedfinishedsquarefeet")
            
    display_distrib("finishedsquarefeet12")            
    train_df["finishedsquarefeet12"] = np.log1p(train_df["finishedsquarefeet12"])
    display_distrib("finishedsquarefeet12")
            
# do feature engineering: add, select, encode
def process_feature():
    # add features
    
    # select features    
    for feature in train_df:
        if train_df[feature].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[feature].values)) 
            train_df[feature] = lbl.transform(list(train_df[feature].values))
            
    train_y = train_df.logerror.values
    train_X = train_df.drop(["transactiondate", "logerror"], axis=1)
    
    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
    
    featureImportance = model.get_fscore()
    features = pd.DataFrame()
    features['features'] = featureImportance.keys()
    features['importance'] = featureImportance.values()
    features.sort_values(by=['importance'],ascending=False,inplace=True)
    fig,ax= plt.subplots()
    fig.set_size_inches(20,10)
    plt.xticks(rotation=90)
    sns.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h",color="#34495e")

    # encode features
    
    
    
def _process_data():
    # load data
    global train_df
    global prop_df
    train_df = pd.read_csv(train_path, parse_dates=["transactiondate"])
    prop_df = pd.read_csv(properties_path)

    #print(pd_train.head(5))
    #print(pd_prop.head(5))
    
    print('train shape before merging', train_df.shape)
    train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
    print('train shape after merging', train_df.shape)
    
    global parcelid
    parcelid = train_df.parcelid
    train_df.drop(['parcelid'], axis=1, inplace=True)
    
    
    #global logerror
    #logerror = train_df.logerror
    #train_df.drop(['logerror'], axis=1, inplace=True)

    # process the target feature (logerror)
    process_target_feature()

    # remove outliers
    process_outlier()
    
    # fill NAs
    process_NA()
    
    # normalize distribution
    process_skewness()
            
    # do feature engineering: add, select, encode
    process_feature()
    
    
## STEP3: build model
def _build_model():
    pass


## STEP3: train
def _train():
    pass


## STEP4: generate submission
def _generate_submission():
    pass


## main
def main():
    _process_data()
    _build_model()
    _train()
    _generate_submission()
    

if __name__ == "__main__":
    main()
    print('\n\n\nThe end.')
    