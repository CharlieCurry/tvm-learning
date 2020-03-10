import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def load_lable_binary(src):
    filename = src+".txt"
    data = np.loadtxt(filename)
    data = np.array(data).reshape((-1,1))
    median = np.median(data)
    # print(median)
    for i in range(data.shape[0]):
        # print(i)
        # print(data[i][3])
        if data[i][:] >= median:
            data[i][:] = 1
        elif data[i][:] < median:
            data[i][:] = 0
    # print(data[:,3])
    # print(data.shape)
    #print(data)

    y = data[:,:].astype(int)  # 标签
    # scaler = StandardScaler()
    # x_std = scaler.fit_transform(x)  # 标准化
    #print(x_std)
    #print(y)
    #x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=.2)
    # print("x_train shape",x_train.shape)
    # print("x_test shape",x_test.shape)
    # print("y_train shape",y_train.shape)
    # print("y_test shape",y_test.shape)
    return y

def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = [505505.90560927434]
    #print(c_range)
    gamma_range = [1.440246537538758]
    #print(gamma_range)
    # 网格搜索交叉验证的参数范围，cv=10,10折交叉
    param_grid = [{'kernel':['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
    # 训练模型
    gs = grid.fit(x_train, y_train)
    # print(gs.best_index_)
    # print(gs.best_score_)
    # print(gs.best_params_)

    #print(grid.predict(x_test))
    #print(y_test)
    # 计算测试集精度
    score = grid.score(x_train, y_train)
    print('train精度为%s' % score)
    score = grid.score(x_test,y_test)
    print('validation精度为%s' % score)

def myxgb(x_train, x_test, y_train, y_test):

    params = {
        'booster': 'gbtree',
        'objective': 'reg:gamma',
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
    print(ypreds)
    print(y_test)
    print(mean_squared_error(y_test,ypreds))
    print(r2_score(y_test,ypreds))
    #print(y_test)
    #print(ypreds)
    #bn = Binarizer(threshold=0.42444044)
    #ypreds = bn.transform(ypreds.reshape(-1, 1))
    #print("myxgb精度为：",accuracy_score(y_test,ypreds))


if __name__ == '__main__':
    X = np.loadtxt("feas.txt",delimiter=' ')
    Y = np.loadtxt("y_train.txt",delimiter=' ')
    print(X)
    print(X.shape)

    Y = np.array(Y).reshape((-1,1))
    print(Y.shape)
    pca = PCA(n_components=7)
    newX = pca.fit_transform(X)

    print(newX.shape)
    print(pca.explained_variance_ratio_)

    newXmax  = newX.astype('float').max(axis=0)
    newXmin  = newX.astype('float').min(axis=0)
    print("newXmax:",newXmax)
    print("newXmin:",newXmin)

    newX = ((newX-newXmin)/(newXmax-newXmin)).astype('float')
    print(newX)

    #myxgb(newX[:500],newX[500:],Y[:500],Y[500:])

    src = "y_train"
    y = load_lable_binary(src)
    print(y.shape)
    svm_c(newX[:600],newX[600:],y[:600],y[600:])
