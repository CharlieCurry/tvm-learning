'''

主要测试TSSModel类的基本使用
'''
import warnings
warnings.filterwarnings("ignore")
from TSSModel import TSSModel
import numpy as np

input_size = 7
tss = TSSModel(input_size=input_size)
tss.load_data('train/feas1.txt','train/y_train.txt')
tss.pca_minmax_split(tss.X,tss.Y,pca_components=input_size)
print(tss.y_train[:10])
print("在线训练和预测")
tss.TSS_NN_Model_fit(tss.x_train, tss.y_train, tss.x_valiation, tss.y_valiation)
tss.pca_minmax(tss.X,tss.Y,pca_components=input_size)
tss.predict(tss.X, tss.Y)
tss.envaluation(tss.y_prediction,tss.Y)
tss.pca_minmax(tss.x_valiation,tss.y_valiation,pca_components=input_size)
tss.predict(tss.X, tss.Y)
tss.envaluation(tss.y_prediction,tss.Y)

print("加载已训练的模型进行预测")

#tss.predict_from_saved_model(X_test, Y_test)
#tss.envaluation(y_prediction,Y_test)
