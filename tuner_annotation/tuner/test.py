
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
print([x[1] for x in heap_items])

scores = [-1,-2,-3,-4,-5,-6,-7,-8]
points = [1,2,3,4,5,6,7,8]
for s, p in zip(scores, points):
    print(s)
    print(p)

print(np.empty_like(points))

new_points = np.array([12,11,14])
new_scores = np.array([10,9,6])
scores = np.array([8,4,11])
t = 1
print(np.minimum((new_scores - scores), 1))
ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
print(ac_prob)
ac_index = np.random.random(len(ac_prob)) < ac_prob
print(ac_index)
points[ac_index] = new_points[ac_index]
scores[ac_index] = new_scores[ac_index]
print(points[ac_index])
print(scores[ac_index])

#zip(new_scores, new_points)
heap_items = [[15,4],[12,1],[10,6]]
print([x[1] for x in heap_items])


