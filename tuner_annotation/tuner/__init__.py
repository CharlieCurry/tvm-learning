#
"""
A tuner takes a task as input.
It proposes some promising :any:`ConfigEntity`in the :any:`ConfigSpace` and measure them on the real hardware.
Then it proposed the next batch of :any:`ConfigEntity` according to the measure results.
This tuning loop is repeated.
"""
'''
task作为输入
实际去跑一些有希望的配置（ConfigEntity、ConfigSpace）
然后依次结果来选择后续的配置
循环此过程

'''
from . import callback
from .tuner import Tuner
from .gridsearch_tuner import GridSearchTuner, RandomTuner
from .ga_tuner import GATuner
from .xgboost_tuner import XGBTuner
