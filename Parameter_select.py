import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics

train_data = pd.read_csv('all/train.csv')
test_data = pd.read_csv('all/test.csv')

# train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
# target = train_data.SalePrice

out_data = train_data [(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)]
train_data = train_data.drop(out_data.index)
#construct new features
train_data['TotalSF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']
test_data['TotalSF'] = train_data['TotalBsmtSF'] + test_data['1stFlrSF'] + train_data['2ndFlrSF']

train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

target = train_data.SalePrice
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
# target = train_data.SalePrice


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

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_testing_predictors = pd.get_dummies(test_predictors)

train_X, test_X = one_hot_encoded_training_predictors.align(one_hot_encoded_testing_predictors, join='inner', axis=1)
my_imputer = Imputer()
imputer_train_X = my_imputer.fit_transform(train_X)
imputer_test_X = my_imputer.fit_transform(test_X)


if __name__ == '__main__':
    # 网格搜索，超参调优
    # cv_params = {'n_estimators':[300, 350, 400, 450, 500]} # 迭代次数调优
    # cv_params = {'learning_rate':[0.04,0.05,0.055,0.06,0.07]} #学习率参数调优
    # cv_params = {'max_depth':range(4, 10), 'min_child_weight':range(1,6)}
    # cv_params = {'gamma':[0.1,0.2,0.3,0.4,0.5,0.6]}
    # cv_params = {'subsample':[0.5,0.55,0.6,0.65,0.7]}
    # cv_params = {'reg_alpha':range(0,6),'reg_lambda':range(0,6)}
    cv_params = {'n_estimators':[300, 350, 400, 450, 500],'learning_rate':[0.04,0.05,0.055,0.06,0.07],'max_depth':range(4, 10), 'min_child_weight':range(1,6),'gamma':[0.1,0.2,0.3,0.4,0.5,0.6],'subsample':[0.5,0.55,0.6,0.65,0.7]}
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
    model = XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2',cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(imputer_train_X, target)
    evaluate_result = optimized_GBM.grid_scores_
    print("每轮迭代运行结果:{0}".format(evaluate_result))
    print("参数的最佳取值:{0}".format(optimized_GBM.best_params_))
    print("最佳模型得分:{0}".format(optimized_GBM.best_score_))
