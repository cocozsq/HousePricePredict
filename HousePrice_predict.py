import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('all/train.csv')
test_data = pd.read_csv('all/test.csv')
# remove scatter data in GrLivArea
out_data = train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)]
train_data = train_data.drop(out_data.index)
# construct new feature
train_data['TotalSF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']
test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']
# 平滑处理待预测的数据
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

target = train_data.SalePrice

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

cols_with_missing = [ col for col in train_data.columns if train_data[col].isnull().any()]

candidate_train_predictors = train_data.drop(['Id','SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id']+cols_with_missing, axis=1)

low_cardinality_cols = [cname for cname in candidate_train_predictors.columns 
                       if candidate_train_predictors[cname].nunique()<10 and candidate_train_predictors[cname].dtype=='object']
numeric_cols = [ cname for cname in candidate_train_predictors.columns
               if candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

# print(train_predictors.columns)
# print(train_predictors.tail())

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
# print(one_hot_encoded_training_predictors)

one_hot_encoded_testing_predictors = pd.get_dummies(test_predictors)
# print(one_hot_encoded_testing_predictors)

# def get_mae(X, y):
    # return -1*cross_val_score(RandomForestRegressor(50), X, y, scoring='neg_mean_absolute_error').mean()

# predictors_without_categoricals = train_data.select_dtypes(exclude=['object'])

# mae_without_categoricals = get_mae(predictors_without_categoricals, target)

# mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

# print("mae without categoricals :", mae_without_categoricals)
# print("mae one hot encoded :", mae_one_hot_encoded)

train_X, test_X = one_hot_encoded_training_predictors.align(one_hot_encoded_testing_predictors, join='inner', axis=1)
my_imputer = Imputer()
imputer_train_X = my_imputer.fit_transform(train_X)
imputer_test_X = my_imputer.fit_transform(test_X)

other_params = {
    'learning_rate':0.05, # 最佳模型参数得分0.8816
    'n_estimators':350,
    'max_depth':4,
    'min_child_weight':3,
    'seed':0,
    'subsample':0.6,
    'colsample_bytree':0.7,
    'gamma':0.1,
    'reg_alpha':1,
    'reg_lambda':1
    }
Xgb_model = XGBRegressor(**other_params)
Xgb_model.fit(imputer_train_X, target, verbose=False)


predict_price = Xgb_model.predict(imputer_test_X)
predict_price = np.expm1(predict_price)

my_submit = pd.DataFrame({'Id':test_data.Id, 'SalePrice':predict_price})

my_submit.to_csv( 'submission7.csv', index=False )  