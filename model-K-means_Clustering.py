import numpy as np
import random as rd
import matplotlib.pyplot as plt

x1 = np.random.normal(3, 1, 100).reshape(-1, 1)
y1 = np.random.normal(5, 1, 100).reshape(-1, 1)
x2 = np.random.normal(5, 1, 100).reshape(-1, 1)
y2 = np.random.normal(3, 1, 100).reshape(-1, 1)
x3 = np.random.normal(3, 1, 100).reshape(-1, 1)
y3 = np.random.normal(2, 1, 100).reshape(-1, 1)

data1 = np.concatenate((x1, y1), axis=1)
data2 = np.concatenate((x2, y2), axis=1)
data3 = np.concatenate((x3, y3), axis=1)
data = np.concatenate((data1, data2, data3), axis=0)
np.random.permutation(data)

att = np.zeros(300).reshape(-1, 1)  # 0,1,2
mu1 = np.asarray([rd.uniform(-1, 8), rd.uniform(-2, 10)]).reshape(1, 2)
mu2 = np.asarray([rd.uniform(-1, 8), rd.uniform(-2, 10)]).reshape(1, 2)
mu3 = np.asarray([rd.uniform(-1, 8), rd.uniform(-2, 10)]).reshape(1, 2)
mu = np.concatenate((mu1, mu2, mu3), axis=0)

# para:rnd,perc
rnd = 1000
perc = 99
obj = np.zeros(rnd)

for itr in range(rnd):
    tmp = np.zeros_like(mu)
    num = [0, 0, 0]
    for i in range(300):
        dis = [0, 0, 0]
        for j in range(3):
            dis[j] = np.sqrt(np.square(data[i, 0] - mu[j, 0]) + np.square(data[i, 1] - mu[j, 1]))
        k = dis.index(min(dis))

        pos = rd.uniform(0, 100)
        if pos < perc:
            att[i] = k
        elif pos < (100 + perc) / 2:
            att[i] = (k + 1) % 3
        else:
            att[i] = (k - 1) % 3

        k = int(att[i])
        obj[itr] = obj[itr] + min(dis)
        num[k] = num[k] + 1
        tmp[k, 0] = tmp[k, 0] + data[i, 0]
        tmp[k, 1] = tmp[k, 1] + data[i, 1]

    for n in range(3):
        mu[n, 0] = tmp[n, 0] / num[n]
        mu[n, 1] = tmp[n, 1] / num[n]


res0 = np.zeros((num[0], 2))
res1 = np.zeros((num[1], 2))
res2 = np.zeros((num[2], 2))

for i in range(300):
    if att[i] == 0:
        num[0] = num[0] - 1
        res0[num[0], 0] = data[i, 0]
        res0[num[0], 1] = data[i, 1]
    elif att[i] == 1:
        num[1] = num[1] - 1
        res1[num[1], 0] = data[i, 0]
        res1[num[1], 1] = data[i, 1]
    else:
        num[2] = num[2] - 1
        res2[num[2], 0] = data[i, 0]
        res2[num[2], 1] = data[i, 1]

print(obj)
plt.scatter(res0[..., 0], res0[..., 1], c='b')
plt.scatter(res1[..., 0], res1[..., 1], c='r')
plt.scatter(res2[..., 0], res2[..., 1], c='g')
plt.show()
