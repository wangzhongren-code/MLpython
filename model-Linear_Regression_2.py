# 线性回归实战：循环发电厂线性回归模型建立与求解
# Input：AT，V，AP，RH
# Output：PE
# 利用9568个样本数据，学习给出五个参数

# Step1 数据读取
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ccpp.csv")
print(data.head())  # 前5行数据
print(data.tail())  # 后5行数据

# Step2 划分训练集和测试集
from sklearn.model_selection import train_test_split

X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Step3 运行scikit-learn提供的线性模型
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print('Coefficient: ', model.coef_)
print('Intercept: ', model.intercept_)

# Step4 模型评价
from sklearn import metrics

y_pred = model.predict(X_test)
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Step5 结果可视化
plt.scatter(y, model.predict(X), s=10, color='yellow')
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle=':', color='blue', linewidth=3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression')
plt.show()
