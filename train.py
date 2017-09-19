# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, skew

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
# remove outliers
def display_logerror():
    plt.figure()
    plt.scatter(x = range(train_df.shape[0]), y = np.sort(train_df['logerror'].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('logerror', fontsize=12)
    plt.show()
    
def display_transaction_data():
    plt.figure()
    counts_in_month = train_df['transactiondate'].dt.month.value_counts()
    sns.barplot(x=counts_in_month.index, y=counts_in_month.values, alpha=0.8, color=color[3])
    plt.xticks(rotation='vertical')
    plt.xlabel('Month of transaction', fontsize=12)
    plt.ylabel('Number of transactions', fontsize=12)
    plt.show()

def display_distrib(feature):
    plt.figure()
    sns.distplot(train_df[feature].dropna() , fit=norm);
    (mu, sigma) = norm.fit(train_df[feature].dropna())    
    
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('distribution')
    plt.show()
    
def display_NA_ratio():
    fig, ax = plt.subplots(figsize=(12,12))
    
    missing_df = prop_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.loc[missing_df['missing_count']>0]
    missing_df = missing_df.sort_values(by='missing_count')
    
    ind = np.arange(missing_df.shape[0])
    ax.barh(ind, missing_df.missing_count.values, color='blue') #horizontal rectangles
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()

def display_latitude_longtitude():
    plt.figure()
    sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)
    plt.ylabel('Longitude', fontsize=12)
    plt.xlabel('Latitude', fontsize=12)
    plt.show()

def display_correlation():
    # Let us just impute the missing values with mean values to compute correlation coefficients #
    #mean_values = train_df.mean(axis=0)
    #train_df_new = train_df.fillna(mean_values, inplace=True)
    
    # Now let us look at the correlation coefficient of each of these variables #
    x_cols = [col for col in train_df.columns if col not in ['logerror'] if train_df[col].dtype=='float64']
    
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(train_df[col].values, train_df.logerror.values)[0,1])
    corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
    corr_df = corr_df.sort_values(by='corr_values')
        
    ind = np.arange(len(labels))
    width = 0.9
    fig, ax = plt.subplots(figsize=(12,40))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
    ax.set_xlabel("Correlation coefficient")
    ax.set_title("Correlation coefficient of the variables")
    #autolabel(rects)
    plt.show()
    
def process_target_feature():
    plt.figure()
    sns.distplot(logerror.dropna() , fit=norm);
    (mu, sigma) = norm.fit(logerror.dropna())    
    
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('distribution')
    plt.show()

# fill NAs
def process_NA():
    # compute NA ratio before adding
    na_ratio = (train_df.isnull().sum() / len(train_df)) * 100    
    na_ratio = na_ratio.sort_values(ascending=False)
    print('NA ratio before adding: \n', na_ratio)
    
    # add NA manually
    train_df['airconditioningtypeid'] = train_df['airconditioningtypeid'].fillna(1) # because all are almost 1
    train_df['architecturalstyletypeid'] = train_df['architecturalstyletypeid'].fillna(7) # because all are almost 7
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
    print('NA ratio after adding: \n', na_ratio)
        
    
# remove outliers based on their correlation with the target feature    
def process_outlier():
    #print(train_df.dtypes)
    pass

# normalize distribution
def process_skewness():
    pass

# do feature engineering: add, remove, select, encoding
def process_feature():
    pass

def _process_data():
    # load data
    global train_df
    global prop_df
    train_df = pd.read_csv(train_path, parse_dates=["transactiondate"])
    prop_df = pd.read_csv(properties_path)

    print('train shape before merging', train_df.shape)
    train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
    print('train shape after merging', train_df.shape)
    
    global parcelid
    parcelid = train_df.parcelid
    train_df.drop(['parcelid'], axis=1, inplace=True)
    
    global logerror
    logerror = train_df.logerror
    train_df.drop(['logerror'], axis=1, inplace=True)
        
  
    
    
    
    #print(pd_train.head(5))
    #print(pd_prop.head(5))
        
    #display_logerror()
    #display_transaction_data()
    #display_NA_ratio()
    #display_latitude_longtitude()
    
    #train_df['transaction_month'] = train_df['transactiondate'].dt.month
    #train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
    #print(train_df.head())
    #print(train_df.dtypes)
    #display_correlation()

    # process target feature (logerror)
    process_target_feature()

    # fill NAs
    process_NA()
    
    # remove outliers
    process_outlier()

    # normalize distribution
    process_skewness()
            
    # do feature engineering: add, remove, select, encoding
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
    