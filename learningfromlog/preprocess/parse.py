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

    #filename = "sourcelog/"+src+".log"
    src = 'XGBTuner_matmul5125125121'
    data = parse(src)
    #sort(data)
    src = 'RDTuner_matmul5125125121'
    data = parse(src)
    src = 'GATuner_matmul5125125121'
    data = parse(src)
    src = 'GRTuner_matmul5125125121'
    data = parse(src)


