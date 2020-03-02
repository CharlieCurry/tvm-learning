'''

主要测试TSSModel类的基本使用
'''
from NNTuner.nntuner.TSSModel import TSSModel
import numpy as np
tss = TSSModel()
# 从外部读入
train_data = np.loadtxt('train/train_data.txt')
test_data = np.loadtxt('valiation/valiation_data.txt')
train_x = train_data[:, :27]
train_y = train_data[:, 27:28]
test_x = test_data[:, :27]
test_y = test_data[:, 27:28]
src = 'matmul'

'''
#读取测试集
'''
valiation_data = np.loadtxt('test/' + src + '_val_totalfeatures.txt')
print("test data shape:", valiation_data.shape)
valiation_x = valiation_data[:, :27].astype('float')
valiation_y = valiation_data[:, 27:28].astype('float')



index = np.random.permutation(len(train_x))
print(train_x[index].shape)
print(train_y[index].shape)
print(train_x[index].shape[1])
print(train_y[index].shape[1])
tss.TSS_NN_Model(train_x, train_y, test_x, test_y)

prediction_y = tss.predict(valiation_x, valiation_y)
print(prediction_y)
