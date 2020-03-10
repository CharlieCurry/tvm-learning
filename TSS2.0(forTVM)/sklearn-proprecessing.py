from sklearn.preprocessing import minmax_scale,scale,maxabs_scale,normalize,binarize
import numpy as np
from TSSModel import TSSModel
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
x = np.array([1,-2,3,4,15,6,7,8]).reshape((2,4))
print(x)
print(minmax_scale(x))
print(scale(x))
print(maxabs_scale(x))
print(binarize(x,threshold=5))
a = np.argmin(x)
print(a)
b = np.argmax(x)
print(b)
print(x[a])

input_size = 7
tss = TSSModel(input_size=input_size)
X,Y = tss.load_data('train/feas1.txt','train/y_train.txt')
regr = LinearSVR(random_state=0, tol=1e-5)
regr.fit(X, Y)
print(regr.coef_)
print(regr.intercept_)
prediction = regr.predict(X)
tss.envaluation(prediction,Y)



