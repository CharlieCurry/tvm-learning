'''

主要测试TSSModel类的基本使用
'''
from offlineTSS.TSSModel import TSSModel
from sklearn.decomposition import PCA
import numpy as np
def load_data(datasrc,labelsrc):
    X = np.loadtxt(datasrc,delimiter=' ')
    Y = np.loadtxt(labelsrc,delimiter=' ')
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
    x_train = newX[:500]
    y_train = Y[:500]
    x_valiation = newX[500:]
    y_valiation = Y[500:]
    print("x_train shape:",x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_valiation shape:", x_valiation.shape)
    print("y_valiation shape:", y_valiation.shape)
    return x_train,y_train,x_valiation,y_valiation





if __name__ == '__main__':
    datasrc = "train/feas1.txt"
    labelsrc = "train/y_train.txt"

    x_train, y_train, x_valiation, y_valiation = load_data(datasrc,labelsrc)
    tss = TSSModel(input_size=x_train.shape[1])
    tss.TSS_NN_Model(x_train,y_train,x_valiation,y_valiation)

    prediction_y = tss.predict(x_valiation, y_valiation)
    print(prediction_y[:10])
    print(y_valiation[:10])
