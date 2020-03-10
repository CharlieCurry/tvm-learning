# -- coding: utf-8 --
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import multiprocessing
import logging
import time
import numpy as np
'''
本类对标xgboost的功能和接口
'''
class TSSModel():
    def __init__(self):
        super(TSSModel, self).__init__()

        print(tf.__version__)

        keep_prob = tf.placeholder(tf.float32)
        xs = tf.placeholder(tf.float32, [None, 27])
        ys = tf.placeholder(tf.float32, [None, 1])
        sess = tf.Session()
        self.xs = xs
        self.ys = ys
        self.sess = sess
        self.keep_prob = keep_prob




    def add_layer(self,input,in_size,out_size,activation_function):
        Weight=tf.Variable(tf.random_normal([in_size,out_size]))
        #Weight = tf.truncated_normal([in_size,out_size], stddev = 0.01)
        biases=tf.Variable(tf.zeros([1,out_size]))+0.000000001
        Wx_plus_b=tf.matmul(input,Weight)+biases
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs

    def TSS_NN_Model(self,train_x,train_y,test_x,test_y):
        input_size = train_x.shape[1]
        output_size = train_y.shape[1]
        print("input size:",input_size)
        print("output size",output_size)
        L1=self.add_layer(self.xs,input_size,32,activation_function=tf.nn.tanh)
        L2=self.add_layer(L1,32,32,activation_function=tf.nn.tanh)
        dropped = tf.nn.dropout(L2,self.keep_prob)
        #L3=add_layer(dropped,64,32,activation_function=tf.nn.tanh)
        prediction = self.add_layer(L2,32,output_size,activation_function=None)
        loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.ys-prediction),reduction_indices=[1]))
        #my_loss = tf.reduce_mean(tf.reduce_sum(tf.where(tf.greater(ys,prediction),(ys-prediction)*lossmore,(prediction-ys)*lossless)))
        hubers = tf.losses.huber_loss(self.ys, prediction,delta=2.0)
        hubers_loss = tf.reduce_sum(hubers)
        #train_step=tf.train.AdamOptimizer(0.001).minimize(hubers_loss)
        #train_step = tf.train.AdadeltaOptimizer(0.01).minimize(hubers_loss)
        train_step=tf.train.RMSPropOptimizer(0.0005).minimize(hubers_loss)
        #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(hubers_loss)
        #train_step = tf.train.AdagradDAOptimizer(0.001).minimize(hubers_loss)
        #train_step = tf.train.AdagradOptimizer(0.01).minimize(hubers_loss)
        init=tf.global_variables_initializer()
        self.sess.run(init)
        batch_size = 512
        data_size = len(train_x)
        STEPS = 6001
        for i in range(STEPS):
            start = (i*batch_size)%data_size
            end = min(start + batch_size,data_size)
            #sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            self.sess.run(train_step,feed_dict={self.xs:train_x[start:end], self.ys:train_y[start:end],self.keep_prob: 0.8})
            if i % 2000 == 0:
                print("i=", i, "train_loss=", self.sess.run(hubers_loss, feed_dict={self.xs:train_x,self.ys:train_y,self.keep_prob: 1}))
                print("i=", i, "valiation_loss=", self.sess.run(hubers_loss, feed_dict={self.xs:test_x,self.ys:test_y,self.keep_prob: 1}))
        # prediction_y = sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        # np.savetxt('prediction/valiation_prediction(28800train).txt',prediction_y,fmt="%2.10f")
        # prediction_y = sess.run(prediction,feed_dict={xs:test_data,ys:test_labels})
        # np.savetxt('prediction/valiation_prediction(3200test).txt',prediction_y,fmt="%2.10f")
        #将训练好的模型保存
        saver=tf.train.Saver()
        #saver.save(sess,'net/X_1')
        #print("net already save to net/X_1")
        self.prediction = prediction
        self.hubers_loss = hubers_loss


    def predict(self,valiation_x,valiation_y):

        # saver = tf.train.import_meta_graph('./net/matmulnet_32000.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./net'))
        import time
        cpu_start = time.clock()
        prediction_y = self.sess.run(self.prediction,feed_dict={self.xs:valiation_x,self.ys:valiation_y,self.keep_prob:1})
        cpu_end = time.clock()
        print('cpu:', cpu_end - cpu_start)
        print(prediction_y.shape)
        print("test_loss=",self.sess.run(self.hubers_loss,feed_dict={self.xs:valiation_x,self.ys:valiation_y,self.keep_prob:1}))
        #np.savetxt('prediction/txt/test_prediction'+src+'.txt',prediction_y,fmt="%2.12f")
        return prediction_y



