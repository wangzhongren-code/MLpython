# 1.文本特征提取
print('文本特征提取')

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'I am Peppa pig',
    'This is my little brother George',
    'This is Mommy pig',
    'And this is Daddy pig',
]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())  # 生成特征向量（ndarray）
print(vectorizer.vocabulary_)  # 建立字典并标号，一个字母不作为单词提取

# 2.图像特征提取
print('图像特征提取')

from sklearn import datasets
import matplotlib.pyplot as plt

# digits手写数据库：1700种0-9手写数字图像，每个图像像素为8*8，像素值0-16，白色为0，黑色为16
digits = datasets.load_digits()
print(digits.target[6])
print(digits.images[6])
plt.figure()
plt.imshow(digits.images[6], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# 得到64维特征向量
print('Feature Vector: \n', digits.images[6].reshape(64, 1))

# 3.语音特征提取
# 常用语音特征：梅尔倒谱系数（MFCC）
# 需要利用数字信号处理相关内容
