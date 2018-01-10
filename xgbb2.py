import numpy as np
import xgboost as xgb
import sklearn.datasets as dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X, y = dataset.make_regression(n_samples=200, n_features=2, random_state=0, noise=4.0,
                       bias=100.0)

plt.scatter(X[:,0],X[:,1])
plt.show()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1729)
print(X_train.shape, X_test.shape)

xlf = xgb.XGBRegressor(max_depth=10,#重要参数，每颗树的最大深度，树高越深，越容易过拟合。
                        learning_rate=0.1,
                        n_estimators=10,
                        silent=True,
                        objective='reg:linear',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)
xlf.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)],early_stopping_rounds=100)
preds = xlf.predict(X_test)
print('score is :',xlf.score(X_test,y_test))