import re
import json
import numpy as np
import pandas as pd
def parse(src):
    srcstr = {}
    tile_size = []
    print(src)
    filename = "sourcelog/matmul"+src+".log.tmp"
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
    print(df[:20])
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
    src = '6464128'
    data = parse(src)
    sort(data)
    #classify2(src)

    src = '6412864'
    data = parse(src)
    sort(data)

    src = '1286464'
    data = parse(src)
    sort(data)

'''out:
6464128
(392, 4)
     tile_x  tile_y  tile_z      time
230    16.0     1.0     8.0  0.000023
74      1.0     4.0    64.0  0.000024
80      1.0     8.0    32.0  0.000025
85      1.0    16.0    64.0  0.000026
75      1.0    16.0    32.0  0.000027
50      1.0    16.0    16.0  0.000027
82      1.0     8.0     8.0  0.000028
107    32.0    16.0    64.0  0.000028
72      1.0    16.0     8.0  0.000029
133     4.0    16.0    64.0  0.000029
136     8.0    16.0    64.0  0.000029
44     16.0    16.0    64.0  0.000029
118     8.0     8.0    64.0  0.000031
132     4.0     8.0    64.0  0.000032
129     1.0     8.0     4.0  0.000033
10      4.0     1.0    64.0  0.000033
266    32.0     1.0     4.0  0.000033
264    32.0     1.0     2.0  0.000033
29     64.0     1.0     4.0  0.000034
121     1.0    16.0     4.0  0.000035
6412864
(392, 4)
     tile_x  tile_y  tile_z      time
197    16.0     1.0     2.0  0.000021
194     8.0     1.0     8.0  0.000028
172    16.0     1.0     4.0  0.000031
193    16.0     1.0     8.0  0.000032
198    32.0     1.0     4.0  0.000033
226    32.0     1.0     2.0  0.000033
13     64.0     1.0     4.0  0.000034
1      64.0     1.0     2.0  0.000035
97     64.0     1.0     1.0  0.000042
102    64.0     1.0     8.0  0.000042
161    64.0     1.0    16.0  0.000051
117     1.0     8.0    32.0  0.000059
138     1.0     8.0     8.0  0.000059
137     1.0    16.0    16.0  0.000059
141     1.0    16.0    32.0  0.000060
114     1.0     8.0     4.0  0.000060
86      4.0     1.0     2.0  0.000061
163     1.0    32.0     1.0  0.000064
79      1.0     2.0    16.0  0.000064
135     1.0    32.0    64.0  0.000064
1286464
(392, 4)
     tile_x  tile_y  tile_z      time
92     16.0    16.0    64.0  0.000019
81      8.0    16.0    64.0  0.000019
179    32.0     1.0     1.0  0.000021
101     1.0     8.0    64.0  0.000021
76      8.0     8.0    64.0  0.000022
18      4.0    16.0    64.0  0.000022
150     1.0     4.0    64.0  0.000024
86     32.0    16.0    32.0  0.000024
223    32.0     1.0    64.0  0.000025
65     16.0    16.0    32.0  0.000025
140    16.0     1.0     4.0  0.000026
74      8.0    16.0    32.0  0.000026
267     8.0     1.0    64.0  0.000026
80     16.0     8.0    32.0  0.000027
133    16.0    32.0    32.0  0.000028
134    32.0    32.0    32.0  0.000028
85      4.0     8.0    32.0  0.000028
124     1.0    16.0    64.0  0.000028
67     16.0     4.0    64.0  0.000029
89      4.0    16.0    32.0  0.000029
'''