# 模型评估：交叉验证法（cross validation）
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
print(iris.data.shape)
print(iris.target.shape)

# K-Fold Cross Validation
from sklearn.model_selection import KFold

X = iris.data  # 4个特征
y = iris.target  # 标记（分类3类:0,1,2）
kf = KFold(n_splits=10)  # 10折交叉检验
for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Stratified K-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold

X = iris.data
y = iris.target
skf = StratifiedKFold(n_splits=10)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], y[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
