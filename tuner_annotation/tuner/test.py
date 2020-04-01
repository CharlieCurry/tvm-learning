
'''
test for sa_model_optimizer

'''

import numpy as np


# config_space = 648
# parallel_size = 128
# points = np.array(sample_ints(0, config_space, parallel_size))
# print(points)


num = 128
heap_items = [(float('-inf'), - 1 - i) for i in range(num)]
print(heap_items)
print(type([x[1] for x in heap_items]))

scores = [-1,-2,-3,-4,-5,-6,-7,-8]
points = [1,2,3,4,5,6,7,8]
for s, p in zip(scores, points):
    print(s)
    print(p)
points = np.array([1,2,3])
print("empty like",np.empty_like(points))
new_points = np.array([12,11,14])
new_scores = np.array([10,9,6])
scores = np.array([8,4,11])
t = 1
print(np.minimum((new_scores - scores), 1))
ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
print("ac prob",ac_prob)
print("random",np.random.random(len(ac_prob)))
ac_index = np.random.random(len(ac_prob)) < ac_prob
print("AC index:",ac_index)
print(type(ac_index))
print(ac_index.shape)
points[ac_index] = new_points[ac_index]
scores[ac_index] = new_scores[ac_index]
print(points)
print(scores)

#zip(new_scores, new_points)
heap_items = [[15,4],[12,1],[10,6]]
print([x[1] for x in heap_items])

import random
res = random.sample([1.0,2.1,3.3],2)
print(res)

for r in res:
    print(int(r))

import pandas as pd
feas = np.array([[1,2,3],[4,5,6]])
dataframe = pd.DataFrame(feas,columns=['i','s','c'])
print(dataframe)
narr = dataframe['s'].values
print(type(narr))
print(list(narr))
flag = False
a = 1
while a<8 and not flag:
    print('hello')
    print(a)
    a+=1