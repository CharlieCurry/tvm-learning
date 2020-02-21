import re
import json
import numpy as np
import pandas as pd
srcstr = {}
tile_size = []
src = '102410241024'
filename = "matmul"+src+".log.tmp"
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
print(tile_size)
np.savetxt(src+'.txt',tile_size,fmt="%2.12f")

data_frame = pd.DataFrame(tile_size,columns=['tile_x','tile_y','tile_z','time'])
print(data_frame)
print(data_frame.describe())
df_group = data_frame.groupby('time',as_index=False).min()
print(df_group)
print(np.median(data_frame['time']))
print(np.median(tile_size[:,3]))