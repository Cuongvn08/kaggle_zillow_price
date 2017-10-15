import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
import datetime as dt
from sklearn.model_selection import KFold
import datetime


print('Loading data...')
properties2016 = pd.read_csv('data/kaggle/zillow_prize/properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('data/kaggle/zillow_prize/properties_2017.csv', low_memory = False)
train2016 = pd.read_csv('data/kaggle/zillow_prize/train_2016_v2.csv')
train2017 = pd.read_csv('data/kaggle/zillow_prize/train_2017.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')
train2017[['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']] = np.nan
train = pd.concat([train2016, train2017], axis = 0)
test = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), 
                how = 'left', on = 'ParcelId')
del properties2016, properties2017, train2016, train2017
gc.collect();


print('Memory usage reduction...')
train[['latitude', 'longitude']] /= 1e6
test[['latitude', 'longitude']] /= 1e6

train['censustractandblock'] /= 1e12
test['censustractandblock'] /= 1e12

for column in test.columns:
    if test[column].dtype == int:
        test[column] = test[column].astype(np.int32)
    if test[column].dtype == float:
        test[column] = test[column].astype(np.float32)
      
        
print('Feature engineering...')
train['month'] = (pd.to_datetime(train['transactiondate']).dt.year - 2016)*12 + pd.to_datetime(train['transactiondate']).dt.month
train = train.drop('transactiondate', axis = 1)
from sklearn.preprocessing import LabelEncoder
non_number_columns = train.dtypes[train.dtypes == object].index.values

for column in non_number_columns:
    train_test = pd.concat([train[column], test[column]], axis = 0)
    encoder = LabelEncoder().fit(train_test.astype(str))
    train[column] = encoder.transform(train[column].astype(str)).astype(np.int32)
    test[column] = encoder.transform(test[column].astype(str)).astype(np.int32)
    
feature_names = [feature for feature in train.columns[2:] if feature != 'month']

month_avgs = train.groupby('month').agg('mean')['logerror'].values - train['logerror'].mean()
                             
print('Preparing arrays and throwing out outliers...')
X_train = train[feature_names].values
y_train = train['logerror'].values
X_test = test[feature_names].values

del test
gc.collect();

month_values = train['month'].values
month_avg_values = np.array([month_avgs[month - 1] for month in month_values]).reshape(-1, 1)
X_train = np.hstack([X_train, month_avg_values])

X_train = X_train[np.abs(y_train) < 0.4, :]
y_train = y_train[np.abs(y_train) < 0.4]

kfolds = 4

models = []
kfold = KFold(n_splits = kfolds, shuffle = True)
for i, (train_index, test_index) in enumerate(kfold.split(X_train, y_train)):
    
    print('Training LGBM model with fold {}...'.format(i + 1))
    X_train_, y_train_ = X_train[train_index], y_train[train_index]
    X_valid_, y_valid_ = X_train[test_index], y_train[test_index]
    
    ltrain = lgb.Dataset(X_train_, label = y_train_, free_raw_data = False)
    lvalid = lgb.Dataset(X_valid_, label = y_valid_, free_raw_data = False)
    
    params = {}
    params['metric'] = 'mae'
    params['max_depth'] = 100
    params['num_leaves'] = 32
    params['feature_fraction'] = .85
    params['bagging_fraction'] = .95
    params['bagging_freq'] = 8
    params['learning_rate'] = 0.01
    params['verbosity'] = 0
    
    models.append(lgb.train(params, ltrain, valid_sets = [ltrain, lvalid], 
            verbose_eval=200, num_boost_round=1000))
                  
                  
print('Making predictions and praying for good results...')
X_test = np.hstack([X_test, np.zeros((X_test.shape[0], 1))])
folds = 10
n = int(X_test.shape[0] / folds)

for j in range(folds):
    results = pd.DataFrame()
    
    if j < folds - 1:
            X_test_ = X_test[j*n: (j+1)*n, :]
            results['ParcelId'] = sample_submission['ParcelId'].iloc[j*n: (j+1)*n]
    else:
            X_test_ = X_test[j*n: , :]
            results['ParcelId'] = sample_submission['ParcelId'].iloc[j*n: ]
            
    for month in [10, 11, 12]:
        X_test_[:, -1] = month_avgs[month - 1]
        assert X_test_.shape[1] == X_test.shape[1]
        y_pred = np.zeros(X_test_.shape[0])
        for model in models:
            y_pred += model.predict(X_test_) / kfolds
        results['2016'+ str(month)] = y_pred
        
        
    X_test_[:, -1] = month_avgs[20]
    assert X_test_.shape[1] == X_test.shape[1]
    y_pred = np.zeros(X_test_.shape[0])
    for model in models:
        y_pred += model.predict(X_test_) / kfolds
    results['201710'] = y_pred
    results['201711'] = y_pred
    results['201712'] = y_pred
    
    if j == 0:
        results_ = results.copy()
    else:
        results_ = pd.concat([results_, results], axis = 0)
    print('{}% completed'.format(round(100*(j+1)/folds)))
    
    
print('Saving predictions...')
results = results_[sample_submission.columns]
assert results.shape == sample_submission.shape
#results.to_csv('submission.csv', index = False, float_format = '%.5f')
results.to_csv('sub{}.csv'.format(datetime.now().\
            strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.5f')
print('Done!')
