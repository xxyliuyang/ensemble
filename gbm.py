import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

X,y = datasets.make_regression(n_samples=20000, n_features=2, random_state=0, noise=4.0,bias=100.0)
#定义交叉验证
param_grid = {
    # 'learning_rate':[0.01,0.02,0.05,0.1],
    # 'n_estimators':[1000,2000,3000,4000,5000],
    # 'num_leaves':[128,1024,4096,16384,32768]
    'learning_rate':[0.01,0.02],
    'n_estimators':[1000,2000],
    'num_leaves':[128]
}

estimator = lgb.LGBMRegressor(colsample_bytree=0.8,subsample=0.9,subsample_freq=5)

fit_params = {'categorical_feature':[0,1,2,3,4,5]}

gbm = GridSearchCV(estimator=estimator,param_grid=param_grid,fit_params=fit_params,refit=True)
#训练和输出
gbm.fit(X,y)

print(gbm.cv_results_)
print('///////////////////////')
print(gbm.best_params_)