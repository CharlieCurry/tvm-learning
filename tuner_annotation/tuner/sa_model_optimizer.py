
"""
Cost model optimizer based on simulated annealing
"""

import heapq
import logging
import time

import numpy as np

from ..util import sample_ints
from .model_based_tuner import ModelOptimizer, knob2point, point2knob

logger = logging.getLogger('autotvm')

class SimulatedAnnealingOptimizer(ModelOptimizer):
    """parallel simulated annealing optimization algorithm

    Parameters
    ----------
    task: Task
        The tuning task
    n_iter: int
        The number of iterations of simulated annealing
    temp: float or Array of float
        If is a single float, then use a constant temperature.
        If is an Array, then perform linear cooling from temp[0] to temp[1]
    early_stop: int, optional
        Stop iteration if the optimal set do not change in `early_stop` rounds
    log_interval: int, optional
        Print log every `log_interval` iterations
    """
    def __init__(self, task, n_iter=500, temp=(1, 0), persistent=True, parallel_size=128,early_stop=50, log_interval=50):
        super(SimulatedAnnealingOptimizer, self).__init__()
        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()]  #eg:[5,6,7]代表每维的可能选项,即维度，乘起来就是所有可能的选项
        self.n_iter = n_iter     #什么的迭代次数？
        self.temp = temp   #Tuple元组：可以理解为一个数组，，，temp意为温度？从1降低
        self.persistent = persistent   #语义：？默认为True
        self.parallel_size = min(parallel_size, len(self.task.config_space))  #什么的并行数量？
        self.early_stop = early_stop or 1e9     #谁的早停次数？
        self.log_interval = log_interval   #log间隙数
        self.points = None        #语义：样本空间中点的索引

    def find_maximums(self, model, num, exclusive):
        '''
        关于是否使用pca,在本方法的调用中，发现使用pca会使得该方法执行时间较长find maxmums cost time: 2.671875(pca)：0.2540767192840576(unpca)
        于是也可从update cost time: 2.9926679134368896： 0.6870121955871582看出
        从现象上看是：里面的predict方法“无故”多次执行：predict count: 1002: 133
        关于每个方法的时间量级见笔记
        '''
        #这一个方法里竟然调用了两次predict,而且每次都是（128，-1）规模------》感觉有改进的空间
        #典型调用：maximums = self.model_optimizer.find_maximums(base_model, self.plan_size, self.visited)
        '''
        :param new cost model and do planing for the next `plan_snum: plan_size: int
        The size of a plan. After `plan_size` trials, the tuner will refit a ize` trials.
        :param exclusive:
        '''
        """Find maximum of a cost model
               Note we use cost model to predict GFLOPS, so we should find the maximum
               Parameters
               ----------
               model: CostModel
                   Cost model
               num: int  ----->常见值对应plan_size = 64
                   The number of returned maximum points
               exclusive: set, optional
                   The excluded set of this optimizer. Return results won't include any
                   elements in this set.
               """
        tic = time.time()#tuning起始时间标记
        temp, n_iter, early_stop, log_interval = self.temp, self.n_iter, self.early_stop, self.log_interval

        if self.persistent and self.points is not None:#如果是持续的; 反复出现的 而且 points 不空
            #之后这个条件满足也就是进入这个逻辑
            print("if self.persistent and self.points is not None:")
            points = self.points
        else:
            print("else:!self.persistent and self.points is not None")
            #一开始是随机采样点
            # Sample m different integer numbers from [low, high) without replacement,返回一个list
            # 就是parallel_size = 128一组，知道空间最大值

            #这个空间可以优化！！！！（1）继续filter(2)根据model预测出来的选
            points = np.array(sample_ints(0, len(self.task.config_space), self.parallel_size))
        #以XGBCostModel为例,这里的model就好比一个xgb已训练好的模型：对新采样的点进行预测
        #这里的points是在[0，len(self.task.config_space))之间采样的self.parallel_size个正整数组成的list,
        # 语义上表示在样本空间特征数据（配置样本）的索引


        scores = model.predict(points)#---->xgboost_cost_model.py
        #scores[i]代表points[i]索引对应的配置通过模型预测到的gflops,所以其shape为：(len(points),)

        # build heap and insert initial points
        #[(-inf, -1), (-inf, -2), (-inf, -3), (-inf, -4),...,(-inf, -63), (-inf, -64)]
        heap_items = [(float('-inf'), - 1 - i) for i in range(num)]
        print("heap_items:",heap_items)
        #建一个堆:堆的初始化，里面包含64和节点对（scores,points）初始化为负数
        heapq.heapify(heap_items)

        #exclusive语义上对应的是已经实际跑过的配置,所以在寻找下次配置时就需要排除考虑
        in_heap = set(exclusive)#set是一个无序不重复的序列，这是一个集合
        print("in_heap:",in_heap)
        #[x[1] for x in heap_items] = [-1,-2,...-63,-64]
        in_heap.update([x[1] for x in heap_items])#Update a set with the union of itself and others

        for s, p in zip(scores, points):#zip之后成了2列的矩阵，s和p分别代表这两列
            if s > heap_items[0][0] and p not in in_heap:  #第一个条件s>-inf肯定满足 ； “p not in in_heap”语义表示采样的索引没有在堆中
                #满足条件，说明这个配置是需要进一步考虑的，所以加入到堆中
                #Pop and return the current smallest value, and add the new item.
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)


        k = 0
        k_last_modify = 0
        ##代码重复？怎么重构------》改进点
        ############################################这个模拟退火的必要性#########################################################
        #temp一般为(1,0)tuple
        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]  #1？
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)#  1  /  (n_iter + 1)？  n_iter=500
        else:
            t = temp
            cool = 0
        # k = 0
        # k_last_modify = 0
        # early_stop = 50
        # n_iter = 500
        while k < n_iter and k < k_last_modify + early_stop:#保证不超过迭代次数
            #Return a new array with the same shape and type as a given array.(数值随机正整数？)
            new_points = np.empty_like(points)
            for i, p in enumerate(points):
                #random walk as local transition；and return new neighborhood index（纯随机）
                new_points[i] = random_walk(p, self.dims)

            #function:筛选出new_points中已经访问过的点,然后补全到和points的数目一致的大小


            #改进点：立马产生的新点可能是已经放入in_heap中，即考虑过的，即先判断是否已经考虑过了，再去predict!!!!
            #其实随机到相同的几率很小，所以这里之前没有这么做，不管怎样，改了试试才知道
            #这里有个问题就是新产生的数目要和之前的一致，如果要去除已经考虑了的，也要考虑到这一点


            #去预测随机的索引对应的配置的性能表现
            #在这个循环里面的predict占用了太多资源
            new_scores = model.predict(new_points)
            #数值上这个变量大于0,----->test.py
            ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))#模拟退火超参数？温度变化？
            '''
            新值比旧值至少好出α（=1e-5）秒，才接受前一项
            否则就是接受1，ac_prob 的取值在(0,e]之间
            '''
            #返回boolean列表------>test.py
            ac_index = np.random.random(len(ac_prob)) < ac_prob##继续增加随机性，随机产生同尺寸的(0,1)之间的浮点数，满足这个条件即视为接受这样的index
            points[ac_index] = new_points[ac_index]
            scores[ac_index] = new_scores[ac_index]

            #这段代码和上面的逻辑一致
            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:#保证考虑过的不放入堆中！！！
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)
                    k_last_modify = k

            k += 1
            t -= cool#哪里有用这个t和cool
            #################################################################################################
            
            #控制log输出和时间输出的
            if log_interval and k % log_interval == 0:
                t_str = "%.2f" % t
                logger.debug("SA iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\ttemp: %s\t"
                             "elapsed: %.2f",
                             k, k_last_modify, heap_items[0][0],
                             np.max([v for v, _ in heap_items]), t_str,
                             time.time() - tic)
        #上面这个while循环结束

        
        heap_items.sort(key=lambda item: -item[0])  #堆排序，便于从中选择性能scores最大？的键值对(其实就是列数为2的矩阵)
        heap_items = [x for x in heap_items if x[0] >= 0]  #从堆中取出所有大于0的可以当作索引的值，作为一个list返回
        logger.debug("SA iter: %d\tlast_update: %d\telapsed: %.2f",
                     k, k_last_modify, time.time() - tic)
        logger.debug("SA Maximums: %s", heap_items)

        if self.persistent:
            self.points = points  #记录到属性中，可以用self(SimulatedAnnealingOptimizer).points?

        return [x[1] for x in heap_items]  #按照出堆的顺序排序(scores==x[0])的列表,取出就是对应的points(=x[1])列表------>test.py

def random_walk(p, dims):
    """random walk as local transition
    Parameters
    ----------
    p: int
        index of the ConfigEntity
    dims: Array of int
        sizes of each dimension

    Returns
    -------
    new_p: int
        new neighborhood index
    """
    # transform to knob form
    #"""convert point form (single integer) to knob form (vector)"""
    old = point2knob(p, dims)
    new = list(old)

    # mutate转变; 转换;
    while new == old:
        #只要相等即随机
        from_i = np.random.randint(len(old))
        to_v = np.random.randint(dims[from_i])
        new[from_i] = to_v
    #出循环就表示产生了新的值
    # transform to index form
    return knob2point(new, dims)
