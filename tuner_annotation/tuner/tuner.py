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
# pylint: disable=unused-argument, no-self-use, invalid-name
"""Base class of tuner"""
import logging

import numpy as np

from ..measure import MeasureInput, create_measure_batch

from ..env import GLOBAL_SCOPE

logger = logging.getLogger('autotvm')

class Tuner(object):
    """Base class for tuners

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task
    """

    def __init__(self, task, **kwargs):
        self.param = kwargs
        self.recorder = None

        self.task = task

        # keep the current best
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None
        self.next_batch_count = 0

    def has_next(self):
        """Whether has next untried config in the space

        Returns
        -------
        has_next: bool
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """get the next batch of configs to be measure on real hardware

        Parameters
        ----------
        batch_size: int
            The size of the batch

        Returns
        -------
        a batch of configs
        """
        raise NotImplementedError()

    def update(self, inputs, results):
        """Update parameters of the tuner according to measurement results

        Parameters
        ----------
        inputs: Array of autotvm.measure.MeasureInput
            The input for measurement
        results: Array of autotvm.measure.MeasureResult
            result for measurement
        """


    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=()):
        """Begin tuning！！！！！Tuner类的入口

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, 'n_parallel', 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0
        while i < n_trial:

            if not self.has_next():
                break
            '''get the next batch of configs to be measure on real hardware'''
            #next_batch的用法
            if self.next_batch_count < min(n_parallel, n_trial - i)//8:
                print("next batch filter")
                configs = self.next_batch_filter(min(n_parallel, n_trial - i))#int batch_size = min(n_parallel, n_trial - i))
                self.next_batch_count +=1
                print("self.next_batch_count",self.next_batch_count)
            else:
                print("next batch")
                configs = self.next_batch(min(n_parallel, n_trial - i))#int batch_size = min(n_parallel, n_trial - i))
            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            #
            #print("tuner inputs:",inputs)
            #
            results = measure_batch(inputs)
            #
            #print("tuner results:",results)
            #

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    #这种优化参数不会导致错误就算它的flops
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    #如果有错，那么其性能就置为0,并记录出错次数
                    flops = 0
                    error_ct += 1

                if flops > self.best_flops:
                    #记录当前最性能和参数配置，迭代次数等信息
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k  #记录真实的迭代轮次，不是指的index

                logger.debug("No: %d\tGFLOPS: %.2f/%.2f\tresult: %s\t%s",i + k + 1, flops / 1e9, self.best_flops / 1e9,res, config)

            i += len(results)  #更新下一次i的迭代起点
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i  #语义：剩下的配置数

            self.update(inputs, results)#----->model_base_tuner.py 和 ga 有实现(重要)
            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Now is in debug mode")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        """load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (MeasureInput, MeasureResult) pair
            Previous tuning records
        """
        raise NotImplementedError()
