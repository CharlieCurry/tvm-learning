import numpy as np
import pickle
import sys
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
def load_data(src):
    filename = src+".txt"
    data = np.loadtxt(filename)
    median = np.median(data[:, 3])
    print(median)
    for i in range(data.shape[0]):
        # print(i)
        # print(data[i][3])
        if data[i][3] >= median:
            data[i][3] = 1
        elif data[i][3] < median:
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
    c_range = [505505.90560927434]
    print(c_range)
    gamma_range = [1.440246537538758]
    print(gamma_range)
    # 网格搜索交叉验证的参数范围，cv=10,10折交叉
    param_grid = [{'kernel':['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
    # 训练模型
    gs = grid.fit(x_train, y_train)
    print(gs.best_index_)
    print(gs.best_score_)
    print(gs.best_params_)

    #print(grid.predict(x_test))
    #print(y_test)
    # 计算测试集精度
    score = grid.score(x_test,y_test)
    print('精度为%s' % score)

    print("-------------------wide testing------------------")
    testsrc = "512512512.txt"
    score = grid.score(*load_test(testsrc))
    print('test的精度为%s' % score)

    testsrc = "102410241024.txt"
    score = grid.score(*load_test(testsrc))
    print('test的精度为%s' % score)

    testsrc = "22422464.txt"
    score = grid.score(*load_test(testsrc))
    print('test的精度为%s' % score)

    testsrc = "102210221022.txt"
    score = grid.score(*load_test(testsrc))
    print('test的精度为%s' % score)

    testsrc = "100010001000.txt"
    score = grid.score(*load_test(testsrc))
    print('test的精度为%s' % score)

def load_test(testsrc):
    test_data = np.loadtxt(testsrc)
    print(testsrc)
    median = np.median(test_data[:, 3])

    for i in range(test_data.shape[0]):
        #print(i)
        #print(data[i][3])
        if test_data[i][3]>= median:
            test_data[i][3]=1
        elif test_data[i][3]< median:
            test_data[i][3]=0
    #print(test_data[:,3])
    print(test_data.shape)
    #print(data)
    x = test_data[:,0:3]#数据特征
    y = test_data[:,3].astype(int)#标签
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)#标准化
    test_x = x_std[:400,:]
    test_y = y[:400]
    print(test_x.shape)
    return test_x,test_y


def myxgb(x_train, x_test, y_train, y_test):

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'gamma': 0.1,
        'max_depth': 8,
        'alpha': 0,
        'lambda': 0,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'min_child_weight': 3,
        'silent': 0,
        'eta': 0.03,
        'nthread': -1,
        'seed': 2019,
    }
    num_round = 180
    dtrain = xgb.DMatrix(x_train, y_train)
    bst = xgb.train(params,dtrain,num_round)
    #pickle.dump(bst,open("xgboostclass2.dat","wb"))
    dtest = xgb.DMatrix(x_test,y_test)
    #loaded_model = pickle.load(open("xgboostclass2.dat","rb"))
    ypreds = bst.predict(dtest)
    #print(y_test)
    #print(ypreds)
    bn = Binarizer(threshold=0.42444044)
    ypreds = bn.transform(ypreds.reshape(-1, 1))
    print(accuracy_score(y_test,ypreds))

if __name__ == '__main__':
    src = '512512512'
    svm_c(*load_data(src))
    myxgb(*load_data(src))