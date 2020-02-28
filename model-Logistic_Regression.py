# Logistic Regression
# 一般用于二分类问题，也用于k分类问题，进行k-1次logistic回归
# 本程序利用logistic回归进行手写数字识别，使用scikit-learn内置数据集digits

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import trange

# 参数设置：
train_size = 0.7  # 统一划分的训练集大小，介于0-1之间
eta = 0.07  # 梯度下降法调整步长
maxCycles = 100  # 梯度下降法迭代步数
Rounds = 100  # 针对每组参数运算次数以取得平均值


def initialization(X, y, num, train_size):
    # X：输入矩阵，y：输出标记（包括所有的），num：当前分类数字
    # 输出分类num的训练集，测试集
    y_num = np.zeros_like(y)
    for i in range(1797):
        if y[i] == num:
            y_num[i] = 1

    return train_test_split(X, y_num, train_size=train_size, test_size=1 - train_size, random_state=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradDescent(dataArr, labelArr, eta, maxCycles):
    # dataArr：输入矩阵，labelArr：输出标记（两者行数相同，均为numpy数组，注意为训练样本），eta：步长，maxCycles：迭代步数
    # 输出回归系数
    X = np.mat(dataArr)  # m*n
    y = np.mat(labelArr)  # m*1
    m, n = np.shape(X)
    weight = np.ones((n, 1))

    for i in range(maxCycles):
        h = sigmoid(X * weight)
        err = y - h
        weight = weight + eta * X.transpose() * err

    return weight


def classify(dataArr, labelArr, weight):
    # dataArr：输入矩阵，labelArr：输出标记（同上，注意为测试样本），weight：训练所得系数
    # 输出测试集上分类准确率
    X = np.mat(dataArr)
    y = np.mat(labelArr)
    h = sigmoid(X * weight)
    m = len(h)
    acc = 0

    for i in range(m):
        if h[i] > 0.5:
            if y[i] == 1:
                acc += 1
        else:
            if y[i] == 0:
                acc += 1

    return acc / m


# 主程序
digits = datasets.load_digits()
X = digits.images.reshape(1797, 64)
y = digits.target.reshape(-1, 1)

mean_acc = np.zeros((1, 10))

for k in trange(Rounds):
    acc = np.zeros((1, 10))
    lis = []
    for i in range(10):
        lis.append(i)

    while True:
        length = len(lis)
        if length == 1:
            acc[0, lis[0]] = acc[0, num]
            break
        num = lis[np.random.randint(length)]
        X_train, X_test, y_train, y_test = initialization(X, y, num, train_size)
        weight = gradDescent(X_train, y_train, eta, maxCycles)
        acc[0, num] = classify(X_test, y_test, weight)
        lis.remove(num)

    mean_acc += acc

mean_acc /= Rounds
mean_acc_all = np.mean(mean_acc)

# 结果可视化：平均精度
xbar = np.arange(10)
ybar = mean_acc.reshape(10)
plt.style.use('ggplot')
plt.bar(xbar, ybar, width=0.5, facecolor='yellowgreen', edgecolor='white')
plt.xticks(np.arange(10))
plt.yticks(np.linspace(0, 1, 11))
plt.xlabel('Digits', fontstyle='italic')
plt.ylabel('Accuracy', fontstyle='italic')
plt.title('Digits Recognition by Logistic Regression', fontstyle='italic')
for x, y in zip(xbar, ybar):
    plt.text(x, y, '%0.4f' % y, ha='center', va='bottom', fontsize='small', color='green')
plt.plot(np.linspace(-1, 10, 100), np.linspace(mean_acc_all, mean_acc_all, 100),
         linestyle='--', linewidth=0.5, color='darkgreen', label='avg = %0.4f' % mean_acc_all)
plt.legend(loc=4)
plt.show()

# 各种style:
# ‘bmh’, ‘classic’, ‘dark_background’, ‘fast’, ‘fivethirtyeight’, ‘ggplot’, ‘grayscale’, ‘seaborn-bright’,
# ‘seaborn-muted’, ‘seaborn-notebook’, ‘seaborn-paper’, ‘seaborn-pastel’, ‘seaborn-poster’, ‘seaborn-talk’,
# ‘seaborn-ticks’, ‘seaborn-white’, ‘seaborn-whitegrid’, ‘seaborn’, ‘Solarize_Light2’, ‘tableau-colorblind10’,
# ‘_classic_test’
