import numpy as np
import matplotlib.pyplot as plt
src ='512512512'
filename = src+".txt"
data = np.loadtxt(filename)
print(data)
print(data.shape)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=666)  # 将数据集分为2类
X = data[:,0:3]

y_pre = km.fit_predict(X)
print(y_pre[:])  # [0 1 1 0 1] 将X 每行对应的数据 为y_pre 类

from sklearn.metrics import calinski_harabaz_score
plt.scatter(X[:, 0], X[:, 1], c=y_pre)
print(calinski_harabaz_score(X, y_pre))
plt.show()