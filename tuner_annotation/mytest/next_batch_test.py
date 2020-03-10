import numpy as np

def next_batch(batch_size):  # batch_size一般为64或128
    ret = []
    plan_size = 64
    trial_pt = 0
    counter = 0
    greedy_counter = 0
    while counter < batch_size:
        if len(visited) >= len(space):
            break
        # self.trials 就是下次要实验的points indexs列表，一般长度为64
        # self.trial_pt初值为0
        while trial_pt < len(trials):
            index = trials[trial_pt]  # 这里的用法很重要
            if index not in visited:    #跳过被访问过的
                break
            trial_pt += 1
        # 这个while循环做完就是 条件为self.trial_pt = len(self.trials)的时候，即此时的trail_pt数目即为还没有被visited过的trials索引的数目
        # 假设在64个中有5个被访问过了，那么限制的trail_pt应该为59

        #print("trail_pt:",trial_pt)
        #这个0.05值越小trails保留得越多，越大新随机得值就越大
        if trial_pt >= len(trials) - int(0.05 * plan_size):  # if 59 >= 64 - 0.05*64?》if 59>=61:
            #print("e-greedy")
            # the tuner is doing the last 5% trials (e-greedy), choose randomly
            index = np.random.randint(len(space))  # 随机产生5%*plan size = 0.05*64 = 3
            while index in visited:  # 如果已经访问过了就再随机，知道随机到没有考虑过的index
                index = np.random.randint(len(space))
            greedy_counter+=1

        ret.append(space[index])  # 将新的index追加到ret中
        visited.add(index)  # 同时标记index为已经考虑过的配置了
        counter += 1
        '''
        改进点:greedy的多少是否代表了一些东西？
        '''
    print("greedy_counter",greedy_counter)
    return ret


if __name__ == '__main__':
    '''
    next_batch:初始情况
    visited set is null
    space large and all can be selected
    '''
    print("##########situation 1-initial############")
    space = [i for i in range(0, 1024, 1)]
    print(len(space))
    visited = set([])
    trials = []  # 通过find maxmumus得来的
    print(len(trials))
    ret = next_batch(64)
    print(ret)
    print(len(ret))

    #中间情况：64个candidate，要是没有访问过的，并且在样本空间中的
    '''
    空间候选还很大
    visited存在一部分
    
    '''
    print("############situation 2-interval######################")
    space = [i for i in range(0,1024,1)]
    print(len(space))
    visited = set([i for i in range(256,1024,2)])
    trials = [i for i in range(128,256,2)] #通过find maxmumus得来的
    print(len(trials))
    ret = next_batch(64)
    print(ret)
    print(len(ret))

    print("############situation 3-almost ending######################")
    space = [i for i in range(0, 1024, 1)]
    print(len(space))
    visited = set([i for i in range(0, 1000, 1)])
    trials = [i for i in range(0, 64, 1)]  # 通过find maxmumus得来的
    print(len(trials))
    ret = next_batch(64)
    print(ret)
    print(len(ret))


'''
关于next_batch的两个trick:
situation 1 initial:不要完全随机的产生,从先验的性能较好的点出发可能会得到更好的效果

situation 3 almost ending:假设这时模型已经能较好的拟合性能曲线了,
那么反复的find_maxmums都是集中在已经访问过的点,那么我们是否可以考虑尽早的结束tuning


'''