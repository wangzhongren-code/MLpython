# sklearn绘制ROC曲线
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  # 导入ROC曲线绘制包与AUC
from sklearn.model_selection import StratifiedKFold

# 导入iris数据集：4特征，3类（0，1，2）
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]  # 提取label不为2的
n_samples, n_features = X.shape  # 元组返回方式：a, b = (x, y)

# 加噪声
'''
补充a：
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
np.r_[a,b] 得到 [1 2 3 4 5 6]
np.c_[a,b] 得到
[[1 4]
 [2 5]
 [3 6]]

补充b：
np.random.rand(M,N) 产生[0,1)均匀分布的M*N随机数矩阵；
np.random.randn(M,N) 产生标准正态分布的M*N随机数矩阵；
np.random.randint(low,high,size=(M,N)默认1个,dtype=默认np.int) 产生[low,high)均匀分布的M*N随机整数矩阵，如果high=None则产生[0,low)；
np.random.uniform(low,high,size=(M,N)默认1个) 产生[low,high)均匀分布的M*N随机浮点数矩阵；
np.random.seed(n) 设置随机数种子，当我们设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数（python默认）。
'''
random_state = np.random.RandomState(0)  # 设置随机数种子始终为0，这样每次运行程序产生相同的随机数
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]  # 产生(100,804)含噪声矩阵

# 绘制ROC曲线
cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)  # np.linspace(start,stop,num)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])  # cyan青色，indigo靛蓝色
lw = 2
i = 0
for (train, test), color in zip(cv.split(X, y), colors):
    model = classifier.fit(X[train], y[train])
    probas_ = model.predict_proba(X[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)  # 进行插值
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=lw, color=color, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))  # %0.2f指保留两位
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', linewidth=lw, color='black', label='Luck')
mean_tpr /= cv.get_n_splits(X, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='green', linestyle='--', label='Mean ROC (AUC = %0.2f)' % mean_auc, linewidth=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc=4)
plt.show()
