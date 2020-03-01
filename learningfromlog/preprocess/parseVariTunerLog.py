import re
import json
import numpy as np
import pandas as pd
def parse(src):
    srcstr = {}
    tile_size = []
    filename = "sourcelog/"+src+".log"
    fp = open(filename,"r")
    for line in fp.readlines():
        line = line.strip()
        #print(line)
        user_dict = json.loads(line)
        #print(user_dict['i'][5]['e'])
        dic = user_dict['i'][5]['e']
        time = user_dict['r'][0][0]
        #print(len(dic))
        for i in range(len(dic)):
            #print(dic[i])
            tile_size=np.append(tile_size,dic[i][2][1])
        tile_size = np.append(tile_size,time)
        tile_size=tile_size.reshape((-1,4))
    print(tile_size.shape)
    np.savetxt('handlelog/'+src+'.txt',tile_size,fmt="%2.12f")
    return tile_size

def sort(data):
    data_frame = pd.DataFrame(data,columns=['tile_x','tile_y','tile_z','time'])
    #print(data_frame.describe())
    # print(data_frame)
    df = data_frame.sort_values(by="time", ascending=True)
    print(df[:10])
    #df_group = data_frame.groupby('time',as_index=False).min()
    #print(df_group)


def classify2(src):
    filename = "handlelog/"+src + ".txt"
    data = np.loadtxt(filename)
    median = np.median(data[:, 3])
    # print(data)
    for i in range(data.shape[0]):
        # print(i)
        # print(data[i][3])
        if data[i][3] >= median:
            data[i][3] = 1
        elif data[i][3] < median:
            data[i][3] = 0
    print(data)


if __name__ == '__main__':
    src = 'Gatuner_matmul512512512'
    data = parse(src)
    sort(data[:50,:])
    #classify2(src)

    src = 'Gruner_matmul512512512'
    data = parse(src)
    sort(data[:50,:])

    src = 'Rtuner_matmul512512512'
    data = parse(src)
    sort(data[:50,:])

    src = 'XGBtuner_matmul512512512'
    data = parse(src)
    sort(data[:50,:])


'''out:
(1000, 4)
     tile_x  tile_y  tile_z      time
385    32.0    16.0    32.0  0.005716
612    16.0    16.0    64.0  0.005993
470    32.0    16.0    64.0  0.006228
361     8.0    16.0    64.0  0.006241
188     8.0    16.0    32.0  0.006291
242    16.0    16.0    32.0  0.006551
468    32.0    32.0    32.0  0.006775
383    32.0     8.0    32.0  0.007002
477   128.0    16.0    64.0  0.007106
648    64.0    16.0    64.0  0.007132
(1000, 4)
     tile_x  tile_y  tile_z      time
545    32.0    16.0    32.0  0.006061
544    16.0    16.0    32.0  0.006579
543     8.0    16.0    32.0  0.006759
645    32.0    16.0    64.0  0.006899
644    16.0    16.0    64.0  0.007190
535    32.0     8.0    32.0  0.007330
643     8.0    16.0    64.0  0.007335
555    32.0    32.0    32.0  0.007402
334    16.0     8.0     8.0  0.007500
534    16.0     8.0    32.0  0.007609
(1000, 4)
     tile_x  tile_y  tile_z      time
930    32.0    16.0    32.0  0.006002
802     8.0    16.0    32.0  0.006510
897    16.0    16.0    32.0  0.006534
193    32.0     8.0    32.0  0.007019
648    32.0    16.0    64.0  0.007082
64      8.0    16.0    64.0  0.007136
184    16.0    16.0    64.0  0.007266
153    16.0    32.0    32.0  0.007384
483    32.0    32.0    32.0  0.007532
418    32.0     8.0    16.0  0.007606
(1000, 4)
     tile_x  tile_y  tile_z      time
134    32.0    16.0    64.0  0.005515
112    32.0    16.0    32.0  0.005956
93     16.0    16.0    64.0  0.006090
17      8.0    16.0    64.0  0.006174
73     16.0    16.0    32.0  0.006401
199   128.0    16.0    64.0  0.006543
72      8.0    16.0    32.0  0.006673
142    32.0    32.0    32.0  0.006788
133    16.0    32.0    32.0  0.006880
74     32.0     8.0    32.0  0.006991
分析：
1.光是看tile参数的变化发现其实不同的tuner在全空间肯定最终时间展示出来差异不大（实际情况下不可能遍历全空间）
2.发现32 32 32的表现的确也不差    
3.在对数据的前50、100的记录排序后发现，即仅在50、100次迭代编译后的结果中发现xgb能更接近更优的参数和性能表现
(1000, 4)
    tile_x  tile_y  tile_z      time
15    32.0     4.0    16.0  0.008626
48     2.0    16.0     8.0  0.012797
32   512.0     1.0     2.0  0.014088
0      4.0     4.0    32.0  0.014265
35     8.0     8.0     2.0  0.016005
24     1.0    32.0     8.0  0.018612
10     1.0    16.0     2.0  0.019296
33     1.0     8.0     2.0  0.020917
21   256.0    32.0    16.0  0.021312
38     1.0    32.0     2.0  0.023831
(1000, 4)
    tile_x  tile_y  tile_z      time
7    128.0     1.0     1.0  0.011567
5     32.0     1.0     1.0  0.012286
6     64.0     1.0     1.0  0.014223
9    512.0     1.0     1.0  0.015599
8    256.0     1.0     1.0  0.018620
32     4.0     8.0     1.0  0.023474
23     8.0     4.0     1.0  0.024989
33     8.0     8.0     1.0  0.025948
35    32.0     8.0     1.0  0.026658
40     1.0    16.0     1.0  0.026854
(1000, 4)
    tile_x  tile_y  tile_z      time
44   128.0     8.0     8.0  0.009138
27    16.0     1.0     4.0  0.010711
32     4.0     4.0     4.0  0.013710
15   512.0     8.0    16.0  0.017821
29    16.0     4.0   256.0  0.021377
10     8.0     2.0    64.0  0.021613
6     32.0     2.0     2.0  0.021936
22     4.0     8.0     1.0  0.023237
41   128.0    16.0     4.0  0.023462
11     2.0     4.0   128.0  0.023875
(1000, 4)
    tile_x  tile_y  tile_z      time
17     8.0    16.0    64.0  0.006174
31    16.0     8.0    32.0  0.007242
35     4.0    32.0    32.0  0.009324
5      8.0    32.0    16.0  0.009665
38    16.0     4.0    64.0  0.010785
43     2.0    16.0    32.0  0.010893
20     2.0    16.0    64.0  0.010972
15     8.0     4.0    32.0  0.012227
30   128.0    16.0     8.0  0.014557
32     1.0    16.0    16.0  0.014918
4.对于512512512的size的问题，对大多数三位均在64以下，也有第一维很大的tile size???
    假设限制在64*64*64的空间下，即有0 1 2 4 8 16 32 64，即8*8*8=512种可能，那么可以就搜索这512的空间吗？
'''

