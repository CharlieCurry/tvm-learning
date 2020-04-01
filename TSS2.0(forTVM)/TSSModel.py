# -- coding: utf-8 --
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import multiprocessing
import logging
import time
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import minmax_scale,scale,maxabs_scale,normalize,Binarizer
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
'''

'''
def count_time(func):
    def int_time(self, *args, **kwargs):
        start_time = datetime.datetime.now()  # 程序开始时间
        func(self, *args, **kwargs)
        print(func)
        over_time = datetime.datetime.now()  # 程序结束时间
        total_time = (over_time - start_time).total_seconds()
        print('程序共计%s秒' % total_time)
    return int_time

class TSSModel():
    def __init__(self,input_size):
        super(TSSModel, self).__init__()
        print(tf.__version__)

        self.xs = tf.placeholder(tf.float32, [None, input_size],name="xs")
        self.ys = tf.placeholder(tf.float32, [None, 1],name="ys")
        self.sess = tf.Session()
        self.keep_prob = tf.placeholder(tf.float32,name="keep_prob")



    @count_time
    def load_data(self,datasrc,labelsrc):
        '''
        :param datasrc: 特征数据集 .txt     n*m
        :param labelsrc: 标签数据集 .txt    n*1
        :return:
        '''
        X = np.loadtxt(datasrc, delimiter=' ')
        Y = np.loadtxt(labelsrc, delimiter=' ')
        Y = np.array(Y).reshape((-1, 1))
        print("X.shape:",X.shape)
        print("Y.shape:", Y.shape)
        self.X = X
        self.Y = Y

    def observe(self,data):
        print("observe your data")
        dataframe = pd.DataFrame(data)
        print(dataframe)
        print(dataframe.describe())
        print(dataframe[0].value_counts())
    @count_time
    def envaluation(self,prediction,label):
        print("envaluation")
        print(prediction[:10])
        print(label[:10])
        a = np.argmin(prediction)
        print("id:", a)
        print("预测出的最小值：", prediction[a:a + 1])
        print("预测出的最小值对应的真实执行时间", label[a:a + 1])
        # print(X[a:a+1,:])
        b = np.argmin(label)
        print("id:", b)
        print("真实的最小值执行时间：", label[b:b + 1])
        print("真实的最小值对应的预测执行时间：", prediction[b:b + 1])
        # print(X[b:b+1,:])
        self.top_time(label)
        self.top_time(prediction)

    @count_time
    def pca_minmax_split(self,X,Y,pca_components):
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        print("pca variance ratio", pca.explained_variance_ratio_)
        print("X.shape :", X.shape)
        X = minmax_scale(X)
        #Y = minmax_scale(Y)
        x_train, x_valiation, y_train, y_valiation = train_test_split(X, Y, test_size=0.5, random_state=42)
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_valiation shape:", x_valiation.shape)
        print("y_valiation shape:", y_valiation.shape)
        self.x_train = x_train
        self.y_train = y_train
        self.x_valiation = x_valiation
        self.y_valiation = y_valiation

    @count_time
    def split(self,X,Y):
        x_train, x_valiation, y_train, y_valiation = train_test_split(X, Y, test_size=0.5, random_state=42)
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_valiation shape:", x_valiation.shape)
        print("y_valiation shape:", y_valiation.shape)
        self.x_train = x_train
        self.y_train = y_train
        self.x_valiation = x_valiation
        self.y_valiation = y_valiation

    @count_time
    def minmax_split(self,X,Y):
        X = minmax_scale(X)
        Y = minmax_scale(Y)
        x_train, x_valiation, y_train, y_valiation = train_test_split(X, Y, test_size=0.5, random_state=42)
        # print("x_train shape:", x_train.shape)
        # print("y_train shape:", y_train.shape)
        # print("x_valiation shape:", x_valiation.shape)
        # print("y_valiation shape:", y_valiation.shape)
        self.x_train = x_train
        self.y_train = y_train
        self.x_valiation = x_valiation
        self.y_valiation = y_valiation

    @count_time
    def normliza_split(self, X, Y):
        X = normalize(X)
        Y = normalize(Y)
        x_train, x_valiation, y_train, y_valiation = train_test_split(X, Y, test_size=0.1, random_state=42)
        # print("x_train shape:", x_train.shape)
        # print("y_train shape:", y_train.shape)
        # print("x_valiation shape:", x_valiation.shape)
        # print("y_valiation shape:", y_valiation.shape)
        self.x_train = x_train
        self.y_train = y_train
        self.x_valiation = x_valiation
        self.y_valiation = y_valiation

    @count_time
    def maxabs_split(self, X, Y):
        X = maxabs_scale(X)
        Y = maxabs_scale(Y)
        x_train, x_valiation, y_train, y_valiation = train_test_split(X, Y, test_size=0.1, random_state=42)
        # print("x_train shape:", x_train.shape)
        # print("y_train shape:", y_train.shape)
        # print("x_valiation shape:", x_valiation.shape)
        # print("y_valiation shape:", y_valiation.shape)
        self.x_train = x_train
        self.y_train = y_train
        self.x_valiation = x_valiation
        self.y_valiation = y_valiation

    @count_time
    def scale_split(self, X, Y):
        X = scale(X)
        Y = scale(Y)
        x_train, x_valiation, y_train, y_valiation = train_test_split(X, Y, test_size=0.1, random_state=42)
        # print("x_train shape:", x_train.shape)
        # print("y_train shape:", y_train.shape)
        # print("x_valiation shape:", x_valiation.shape)
        # print("y_valiation shape:", y_valiation.shape)
        self.x_train = x_train
        self.y_train = y_train
        self.x_valiation = x_valiation
        self.y_valiation = y_valiation

    @count_time
    def pca_minmax(self,X,Y,pca_components):
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        print("pca variance ratio", pca.explained_variance_ratio_)
        print("X.shape :", X.shape)
        X = minmax_scale(X)
        #Y = minmax_scale(Y)
        self.X = X
        self.Y = Y
    @count_time
    def top_time(self,data):
        data_frame = pd.DataFrame(data, columns=['time'])
        # print(data_frame.describe())
        # print(data_frame)
        df = data_frame.sort_values(by="time", ascending=True)
        print(df[:20])
        # df_group = data_frame.groupby('time',as_index=False).min()
        # print(df_group)

    def add_layer(self,input,in_size,out_size,activation_function):
        Weight=tf.Variable(tf.random_normal([in_size,out_size]))
        #Weight = tf.truncated_normal([in_size,out_size], stddev = 0.01)
        biases=tf.Variable(tf.zeros([1,out_size]))+0.000000001
        Wx_plus_b=tf.add(tf.matmul(input,Weight),biases)
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs

    @count_time
    def TSS_NN_Model_fit(self, x_train, y_train, x_valiation, y_valiation, l1_size=64, l2_size=32, output_size=1, delta=2.0, learning_rate=0.005, batch_size=32, steps=50001, keep_prob=0.8):
        '''

        :param x_train:
        :param y_train:
        :param x_valiation:
        :param y_valiation:
        :param l1_size:
        :param l2_size:
        :param output_size:
        :param delta:
        :param learning_rate:
        :param batch_size:
        :param steps:
        :param keep_prob:
        :return:
        '''

        input_size = x_train.shape[1]
        output_size = y_train.shape[1]
        print("###################tss nn model parameters###################")
        print("input size:",input_size,"l1_size:",l1_size,"l2_size:",l2_size,"output_size:",output_size,"delta:",delta,"learning rate:",learning_rate,"batch size:",batch_size,"steps:",steps,"keep prob:",keep_prob)
        print("#############################################################")
        L1=self.add_layer(self.xs,input_size,l1_size,activation_function=tf.nn.tanh)
        L2=self.add_layer(L1,l1_size,l2_size,activation_function=tf.nn.tanh)
        #dropped = tf.nn.dropout(L2,self.keep_prob)
        #L3=self.add_layer(dropped,l2_size,32,activation_function=tf.nn.tanh)

        self.prediction = self.add_layer(L2,l2_size,output_size,activation_function=None)
        #self.prediction = tf.get_variable('prediction', shape=[None,1], initializer=prediction)
        #prediction = tf.get_variable(name="prediction",initializer=prediction)
        #loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.ys-prediction),reduction_indices=[1]))
        #my_loss = tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(ys,prediction),(ys-prediction)*lossmore,(prediction-ys)*lossless)))

        hubers = tf.losses.huber_loss(self.ys,self.prediction,delta=delta)
        self.hubers_loss = tf.reduce_sum(hubers,name="hubers_loss")
        #train_step=tf.train.AdamOptimizer(0.001).minimize(hubers_loss)
        #train_step = tf.train.AdadeltaOptimizer(0.01).minimize(hubers_loss)
        train_step=tf.train.RMSPropOptimizer(learning_rate).minimize(self.hubers_loss)
        #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(hubers_loss)
        #train_step = tf.train.AdagradDAOptimizer(0.001).minimize(hubers_loss)
        #train_step = tf.train.AdagradOptimizer(0.01).minimize(hubers_loss)
        init=tf.global_variables_initializer()
        self.sess.run(init)
        batch_size = batch_size
        data_size = len(x_train)
        STEPS = steps
        for i in range(STEPS):
            start = (i*batch_size)%data_size
            end = min(start + batch_size,data_size)
            #sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            self.sess.run(train_step, feed_dict={self.xs: x_train[start:end], self.ys: y_train[start:end], self.keep_prob: keep_prob})
            if i % 2000 == 0:
                print("i=", i, "train_loss=", self.sess.run(self.hubers_loss, feed_dict={self.xs:x_train, self.ys:y_train, self.keep_prob: 1}))
        print("valiation_loss=", self.sess.run(self.hubers_loss, feed_dict={self.xs: x_valiation, self.ys: y_valiation, self.keep_prob: 1}))
        #将训练好的模型保存
        saver=tf.train.Saver()
        saver.save(self.sess, './checkpoint_dir/TSSModel')
        print("model saved!")

    def TSS_XGB_Model(self,x_train, y_train, x_valiation, y_valiation):
        print("xgb regression")
    @count_time
    def predict(self, x_test, y_test):
        import time
        cpu_start = time.clock()
        prediction_y = self.sess.run(self.prediction, feed_dict={self.xs:x_test, self.ys:y_test, self.keep_prob:1})
        cpu_end = time.clock()
        print('predict cost cpu time:', cpu_end - cpu_start)
        print(prediction_y.shape)
        print("test hubers loss=", self.sess.run(self.hubers_loss, feed_dict={self.xs:x_test, self.ys:y_test, self.keep_prob:1}))
        #np.savetxt('prediction/txt/test_prediction'+src+'.txt',prediction_y,fmt="%2.12f")
        print("mean_squared_error:", mean_squared_error(prediction_y, y_test))
        self.y_prediction = prediction_y

    @count_time
    def predict_from_saved_model(self, x_test, y_test):
        print("x test shape:",x_test.shape)
        print("y test shape:",y_test.shape)
        import time
        import tensorflow as tf
        sess = tf.Session()
        saver = tf.train.import_meta_graph('./checkpoint_dir/TSSModel.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
        graph = tf.get_default_graph()

        xs = graph.get_tensor_by_name("xs:0")
        ys = graph.get_tensor_by_name("ys:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        #prediction = graph.get_tensor_by_name("outputs:0")
        hubers_loss = graph.get_tensor_by_name("hubers_loss:0")

        cpu_start = time.clock()
        #prediction_y = sess.run(prediction, feed_dict={xs:x_test, ys:y_test, keep_prob:1})
        #print(prediction_y.shape)
        cpu_end = time.clock()
        print('predict cost cpu time:', cpu_end - cpu_start)
        #print(prediction_y.shape)
        print("test hubers loss=", sess.run(hubers_loss, feed_dict={xs:x_test, ys:y_test, keep_prob:1}))
        #np.savetxt('prediction/txt/test_prediction'+src+'.txt',prediction_y,fmt="%2.12f")
        #print("mean_squared_error:", mean_squared_error(prediction_y, y_test))
        #return prediction_y



