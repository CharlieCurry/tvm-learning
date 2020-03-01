from .model_based_tuner import ModelBasedTuner, ModelOptimizer
from .xgboost_cost_model import XGBoostCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer


class NNTuner(ModelBasedTuner):
    def __init__(self, task, plan_size=64,
                 feature_type='itervar', loss_type='rank', num_threads=None,
                 optimizer='sa', diversity_filter_ratio=None, log_interval=50):














    def tune(self, *args, **kwargs):
