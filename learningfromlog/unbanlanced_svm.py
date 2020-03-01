# Load libraries
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
#只加载两个类别的数据，两类，各50个
iris = datasets.load_iris()
X = iris.data[:100,:]
y = iris.target[:100]
# 删掉前四十个数据，数据总数变为60个
X = X[40:,:]
y = y[40:]

# 类别为0的类别不变，类别不为0的全部变为1
y = np.where((y == 0), 0, 1)
print(y)
# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
# Create support vector classifier
svc = SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=0)

# Train classifier
model = svc.fit(X_std, y)

score = model.score(scaler.fit_transform(iris.data[80:100,:]),iris.target[80:100])
print('validation精度为%s' % score)

score = model.score(scaler.fit_transform(iris.data[40:60,:]),iris.target[40:60])
print('validation精度为%s' % score)

score = model.score(scaler.fit_transform(iris.data[10:30,:]),iris.target[10:30])
print('validation精度为%s' % score)

score = model.score(scaler.fit_transform(iris.data[:10,:]),iris.target[:10])
print('validation精度为%s' % score)