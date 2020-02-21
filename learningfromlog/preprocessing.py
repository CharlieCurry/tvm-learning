import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
def load_data(src):
    filename = src+".txt"
    data = np.loadtxt(filename)
    #print(data)
    for i in range(data.shape[0]):
        #print(i)
        #print(data[i][3])
        if data[i][3]>= 4.968840e-03:
            data[i][3] = 1
        elif data[i][3]<  4.968840e-03:
            data[i][3] = 0
    print(data[:,3])
    print(data.shape)
    #print(data)
    x = data[:,0:3]  # 数据特征
    y = data[:,3].astype(int)  # 标签
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)  # 标准化
    #print(x_std)
    #print(y)
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=.2)
    print("x_train shape",x_train.shape)
    print("x_test shape",x_test.shape)
    print("y_train shape",y_train.shape)
    print("y_test shape",y_test.shape)
    return x_train, x_test, y_train, y_test

def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(10, 20, 20, base=2)
    print(c_range)
    gamma_range = np.logspace(-10, 10, 20, base=2)
    print(gamma_range)
    # 网格搜索交叉验证的参数范围，cv=10,10折交叉
    param_grid = [{'kernel':['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
    # 训练模型
    gs = grid.fit(x_train, y_train)
    print(gs.best_index_)
    print(gs.best_score_)
    print(gs.best_params_)

    print(grid.predict(x_test))
    print(y_test)
    # 计算测试集精度
    score = grid.score(x_test,y_test)
    print('精度为%s' % score)

if __name__ == '__main__':
    src = '512512512'
    svm_c(*load_data(src))