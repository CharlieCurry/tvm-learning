# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return,invalid-name,consider-using-enumerate,abstract-method
"""Base class for model-based tuner
This type of tuner will fit a cost model and use some optimization methods to
find optimums points of cost model in space.
"""
import gc

import numpy as np
import time
from .tuner import Tuner
from ..env import GLOBAL_SCOPE

class FeatureCache(object):
    """Feature cache manager for cache sharing between different cost models"""
    def __init__(self):
        self.feature_cache = {}

    def get(self, key):
        """ Get feature cache dictionary for a key

        Parameters
        ----------
        key: str
            The key of a feature type

        Returns
        -------
        fea_cache: dict
            cache dictionary
        """
        if key not in self.feature_cache:
            self.feature_cache[key] = {}

        return self.feature_cache[key]

    def size(self, key):
        """" Get the size of a feature cache dictionary

        Parameters
        ----------
        key: str
            The key of a feature type

        Returns
        -------
        n: int
        """
        return len(self.feature_cache.get(key, tuple()))

    def clear(self, key):
        """Clear feature cache for a key

        Parameters
        ----------
        key: str
            The key of a feature type
        """
        del self.feature_cache[key]
        self.feature_cache[key] = {}
        gc.collect()


class CostModel(object):
    """Cost model to predict the speed of a config"""
    def __init__(self):
        pass

    def fit(self, xs, ys, plan_size):
        """Fit to training data

        Parameters
        ----------
        xs: Array of int
            indexes of configs in the config space
        ys: Array of float
            The speed (flop, float number operations per second)
        plan_size: int
            The plan size of tuner
        """
        raise NotImplementedError()

    def fit_log(self, records, plan_size):
        """Fit training data from log.

        Parameters
        ----------
        records: Array of Tuple(MeasureInput, MeasureResult)!!!!!!!!!
            The tuning records
        plan_size: int
            The plan size of tuner
        """
        raise NotImplementedError()

    def predict(self, xs, output_margin=False):
        """Predict the speed of configs

        Parameters
        ----------
        xs: Array of int
            The indexes of configs to predict
        output_margin: bool, optional
            Whether output the untransformed margin.
            When a model is used as base model, it should output untransformed margin

        Returns
        -------
        preds: Array of float
            The prediction
        """
        raise NotImplementedError()

    def load_basemodel(self, base_model):
        """Load base model for transfer learning

        Parameters
        ----------
        base_model: CostModel
                base model
        """
        raise NotImplementedError()

    def spawn_base_model(self):
        """Clone a base model with the same parameters.
        The base model is used to fit history data in transfer learning.

        Returns
        -------
        model: CostModel
            A model with the same hyperparameter (argument)
        """
        raise NotImplementedError()


class ModelOptimizer(object):
    """Optimizer used to find optimal points of cost model"""
    def __init__(self):
        pass

    def find_maximums(self, model, num, exclusive):
        """Find maximum of a cost model

        Note we use cost model to predict GFLOPS, so we should find the maximum

        Parameters
        ----------
        model: CostModel
            Cost model
        num: int
            The number of returned maximum points
        exclusive: set, optional
            The excluded set of this optimizer. Return results won't include any
            elements in this set.
        """
        raise NotImplementedError()


class ModelBasedTuner(Tuner):
    """Base class for model based tuner
    This type of tuner will fit a cost model and use an optimizer to
    find the maximums of the cost model as next trials

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task
    cost_model: CostModel
        The cost model that predicts the speed of a config (IR)
    model_optimizer:
        The optimizer to find local optimum points of cost model in tuning search space
    plan_size: int
        Tuner will re-fit model per `plan_size` new measure samples
    diversity_filter_ratio: int or float, optional
        If is not None, the tuner will first select
        top-(plan_size * diversity_filter_ratio) candidates according to the cost model
        and then pick plan_size of them according to the diversity metric.
    """

    def __init__(self, task, cost_model, model_optimizer, plan_size, diversity_filter_ratio=None):
        super(ModelBasedTuner, self).__init__(task)

        # space
        self.task = task
        self.target = task.target
        self.plan_size = plan_size
        self.space = task.config_space
        self.space_len = len(task.config_space)
        self.dims = [len(x) for x in self.space.space_map.values()]

        self.cost_model = cost_model
        self.model_optimizer = model_optimizer
        self.diversity_filter_ratio = diversity_filter_ratio

        if self.diversity_filter_ratio:
            assert self.diversity_filter_ratio >= 1, "Diversity filter ratio " \
                                                     "must be larger than one"

        # trial plan
        self.trials = []   #视为一个列表或矩阵
        self.trial_pt = 0  #trail_plan_trails? 语义：？？？？
        self.visited = set()   #已经访问的梵高一个集合里

        # observed samples


        self.xs = []   #矩阵  用法：x_train = self._get_feature(xs)
        self.ys = []   #矩阵  用法：y_train = np.array(ys)
        self.flops_max = 0.0    #记录再tuning过程中最大的flops
        self.train_ct = 0   #这里置0说明作用域有限
        self.rets = []

    def next_batch(self, batch_size):#batch_size=8------>看next_batch_test.py
        tic = time.time()
        ret = []
        counter = 0
        while counter < batch_size:
            if len(self.visited) >= len(self.space):
                break
            #self.trials 就是下次要实验的points indexs列表，一般长度为64
            #self.trial_pt初值为0
            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]  #这里的用法很重要
                if index not in self.visited:
                    break  #这里break意味这下一句不执行
                self.trial_pt += 1
            #这个while循环做完就是 条件为self.trial_pt = len(self.trials)的时候，即此时的trail_pt数目即为还没有被visited过的trials索引的数目
            #假设在64个中有5个被访问过了，那么限制的trail_pt应该为59
            #这个while只会在第一次执行，后面就不满足条件了

            if self.trial_pt >= len(self.trials) - int(0.05 * self.plan_size):#  if 59 >= 64 - 0.05*64?》if 59>=61:
                # the tuner is doing the last 5% trials (e-greedy), choose randomly
                index = np.random.randint(len(self.space))  #随机产生5%*plan size = 0.05*64 = 3
                while index in self.visited:                       #如果已经访问过了就再随机，知道随机到没有考虑过的index
                    index = np.random.randint(len(self.space))

            ret.append(self.space.get(index))   #将新的index追加到ret中
            self.visited.add(index)             #同时标记index为已经考虑过的配置了

            counter += 1
        #
        self.rets.append(ret)
        #
        print("next batch cost time:",time.time()-tic)
        return ret


    def next_batch_filter(self, batch_size):  # batch_size一般为64或128------>看next_batch_test.py
        import re
        import numpy as np
        import pandas as pd
        import random
        tic = time.time()
        filename = "res.txt"
        fp = open(filename, "r")
        ress = []
        for line in fp.readlines():
            line = line.strip()
            pattern1 = re.compile("\d+", re.S)
            data1 = pattern1.findall(line)
            # print(data1)
            res = []
            for i in data1:
                res.append(int(i))
            ress.append(res)
        ress = np.array(ress)
        dataframe = pd.DataFrame(ress, columns=['x', 'fx', 'y', 'fy', 'k', 'fk', 'index'])
        # print(dataframe)
        filterdata = dataframe[(dataframe['fx'] <= 32) & (dataframe['fy'] <= 32) & (dataframe['fk'] <= 32)]
        index = filterdata['index'].tolist()
        print(index)
        print(len(index))
        slice = random.sample(index, batch_size)
        print(slice)
        ret = []
        for i in slice:
            ret.append(self.space.get(i))  # 将新的index追加到ret中
            self.visited.add(i)  # 同时标记index为已经考虑过的配置了
        self.rets.append(ret)
        print("next batch filter cost time:", time.time() - tic)
        self.isFrist = False
        return ret



    def update(self, inputs, results):#最为重要的方法之一
        for inp, res in zip(inputs, results):
            index = inp.config.index

            if res.error_no == 0:
                self.xs.append(index)
                flops = inp.task.flop / np.mean(res.costs)
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
            else:
                self.xs.append(index)
                self.ys.append(0.0)

        # if we have enough new training samples
        #train_ct,记录了当前有多少的配置被考虑的数目？
        if len(self.xs) >= self.plan_size * (self.train_ct + 1) and self.flops_max > 1e-6:
            #调用fit
            self.cost_model.fit(self.xs, self.ys, self.plan_size)

            if self.diversity_filter_ratio:#默认是None,而且下面的0.2是0------->有改进之处
                print("diversity_filter_ratio not None----方式一")
                candidate = self.model_optimizer.find_maximums(self.cost_model, self.plan_size * self.diversity_filter_ratio, self.visited)
                #调用predict
                scores = self.cost_model.predict(candidate)
                knobs = [point2knob(x, self.dims) for x in candidate]
                #调用submodular_pick
                pick_index = submodular_pick(0.2 * scores, knobs, self.plan_size, knob_weight=1)#原本这0.2是0
                maximums = np.array(candidate)[pick_index]
            else:
                print("diversity_filter_ratio default is None----方式二")
                #find_maximums------>sa_model_optimizer
                maximums = self.model_optimizer.find_maximums(self.cost_model, self.plan_size, self.visited)
                #得到的是一个按性能降序排序所对应的points列表

            self.trials = maximums  #这就是下次要实验的points indexs列表
            self.trial_pt = 0       #这里置0了说明trail_pt是每次update控制self.trails列表的索引
            self.train_ct += 1

    def load_history(self, data_set):
        # set in_tuning as True to make the feature extraction consistent
        GLOBAL_SCOPE.in_tuning = True

        # fit base model
        base_model = self.cost_model.spawn_base_model()
        #fit_log
        success = base_model.fit_log(data_set, self.plan_size)

        if not success:
            GLOBAL_SCOPE.in_tuning = False
            return

        # use base model to select initial points
        if not self.trials:
            # no plan yet, use base model to select initial trials
            maximums = self.model_optimizer.find_maximums(base_model, self.plan_size, self.visited)
            self.trials = maximums
            self.trial_pt = 0

        self.cost_model.load_basemodel(base_model)
        GLOBAL_SCOPE.in_tuning = False

    def has_next(self):
        return len(self.visited) < len(self.space)


def point2knob(p, dims):
    """convert point form (single integer) to knob form (vector)"""
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim
    return knob


def knob2point(knob, dims):
    """convert knob form (vector) to point form (single integer)"""
    p = 0
    for j, k in enumerate(knob):
        p += int(np.prod(dims[:j])) * k
    return p


def submodular_pick(scores, knobs, n_pick, knob_weight=1.0):
    """Run greedy optimization to pick points with regard to both score and diversity.

    DiversityScore = knob_weight * number of unique knobs in the selected set
    Obj = sum(scores[i] for i in pick) + DiversityScore
    Note that this objective function is a monotone submodular function.

    Parameters
    ----------
    scores: Array of float
        score of every points
    knobs: Array of Array of int
        feature vector (tunable knobs) of every points
    n_pick: int
        number of points to pick
    knob_weight: float
        weight of an unique knob feature
    """
    n = len(scores)
    assert n == len(knobs)
    n_knobs = len(knobs[0])

    knobs_set = [set() for _ in range(n_knobs)]

    ret = []
    remain = list(range(len(scores)))

    for _ in range(n_pick):
        max_x = -1
        max_delta = -1e9

        for x in remain:
            tmp_delta = scores[x]
            for i in range(n_knobs):
                if knobs[x][i] not in knobs_set[i]:
                    tmp_delta += knob_weight

            if tmp_delta > max_delta:
                max_delta, max_x = tmp_delta, x

        ret.append(max_x)
        remain.remove(max_x)
        for i in range(n_knobs):
            knobs_set[i].add(knobs[max_x][i])

    return ret
