import re
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def parse(filename):
    fp = open(filename,"r")
    ress = []
    for line in fp.readlines():
        line = line.strip('\n')
        #r'-?\d+\.?\d*e?-?\d*?'
        pattern1 = re.compile(r'-?\d+\.?\d*e?-?\d*?', re.S)
        data1 = pattern1.findall(line)
        res = []
        for i in data1:
            res.append(float(i))
        ress.append(res)
    count = 0
    data = []

    for r in ress:
        count += 1
        if count % 2 == 0:
            data.append(r)
    data = np.array(data).reshape((-1,6))
    #print(data)
    print(data.shape)
    return data

def plotdouble(data1, data2):
    columns = ['current', 'best', 'iter', 'total_iter', 'cost_time','batch_time']
    dataframe1 = pd.DataFrame(data1, columns=columns)
    dataframe2 = pd.DataFrame(data2, columns=columns)
    plt.plot(dataframe1['iter'], dataframe1['best'].values,color='r',label='tuner')
    plt.plot(dataframe2['iter'], dataframe2['best'].values,color='b',label='tuner33')
    plt.xlabel('Task 12/12', fontsize=12)
    plt.ylabel('Gflops', fontsize=12)
    plt.legend(loc=0,ncol=1,fontsize=12)
    plt.ylim(0,130)
    plt.show()

def plot_tuner(data,color,label):
    columns = ['current', 'best', 'iter', 'total_iter', 'cost_time','batch_time']
    dataframe1 = pd.DataFrame(data, columns=columns)
    plt.plot(dataframe1['iter'], dataframe1['best'].values, color=color, label=label)




if __name__ == '__main__':

    # filename1 = "tuner0/tuner0base_2_512256256.txt"
    # data1 = parse(filename1)
    # filename2 = "tuner33/tuner33_8_512256256.txt"
    # data2 = parse(filename2)
    # plotdouble(data1, data2)
    # for i in range(6):
    #     filename = "release_test_"+str(i)+"_adaptive.txt"
    #     data = parse(filename)
    #     plot_tuner(data,color='purple',label='release')
    #

    # for i in range(6):
    #     filename = "tuner70_test_128_"+str(i)+".txt"
    #     data = parse(filename)
    #     plot_tuner(data,color='brown',label='tuner70')

    for i in range(6):
        filename = "tuner50_1_test_256_"+str(i)+".txt"
        data = parse(filename)
        plot_tuner(data,color='orange',label='tuner50:reg')

    for i in range(6):
        filename = "tuner51_1_test_128_"+str(i)+".txt"
        data = parse(filename)
        plot_tuner(data,color='r',label='tuner51_reg+sa')
    #
    # for i in range(6):
    #     filename = "tuner51_"+str(i+1)+"_test_128_rank.txt"
    #     data = parse(filename)
    #     plot_tuner(data,'y',label='tuner51_rank+sa')
    #
    # for i in range(6):
    #     filename = "tuner60_1_test_128_"+str(i)+".txt"
    #     data = parse(filename)
    #     plot_tuner(data,'black',label='tuner60+sa+rank+reg+persistent')
    #
    # for i in range(6):
    #     filename = "tuner61_1_test_128_"+str(i+1)+".txt"
    #     data = parse(filename)
    #     plot_tuner(data,'blue',label='tuner61+sa+rank+reg+Non-persistent')
    #
    for i in range(6):
        filename = "tuner0_1_test_128_"+str(i)+".txt"
        data = parse(filename)
        plot_tuner(data,color='g',label='tuner0:rank')




    plt.xlabel('iter', fontsize=12)
    plt.ylabel('Gflops', fontsize=12)
    plt.legend(loc=0, ncol=1, fontsize=4)
    plt.ylim(200, 500)
    plt.show()




