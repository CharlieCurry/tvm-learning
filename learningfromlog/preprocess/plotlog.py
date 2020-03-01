import matplotlib.pyplot as plt
import numpy as np
def plot(src):
    filename = "handlelog/" + src + ".txt"
    data = np.loadtxt(filename)
    #print(data)
    #print(data.shape)
    x = np.arange(data.shape[0])
    #print(x)
    y = data[:, 3]

    #print(y)
    sc = plt.scatter(x, y)
    plt.legend(handles=[sc], labels=[src], loc='upper right')
    plt.ylim(0, 0.5)
    plt.show()

if __name__ == '__main__':

    src = 'XGBTuner_matmul512512512'
    plot(src)
    #
    src = 'GRTuner_matmul512512512'
    plot(src)
    #
    src = 'GATuner_matmul512512512'
    plot(src)
    #
    src = 'RDTuner_matmul512512512'
    plot(src)

    src = 'XGBTuner_matmul5125125121'
    plot(src)
    #
    src = 'GRTuner_matmul5125125121'
    plot(src)
    #
    src = 'GATuner_matmul5125125121'
    plot(src)
    #
    src = 'RDTuner_matmul5125125121'
    plot(src)
    # src = '1286464'
    # plot(src)
    # src = '6412864'
    # plot(src)
    # src = '6464128'
    # plot(src)

#观测结论：xgb有明显收敛的表现,在四个tuner的性能和耗时都算上游