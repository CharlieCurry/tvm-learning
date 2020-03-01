"""Tuner that uses xgboost as cost model"""

from .model_based_tuner import ModelBasedTuner, ModelOptimizer
from .nn_cost_model import NNCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer

class NNTuner(ModelBasedTuner):
    def __init__(self, task, plan_size=64,
                 feature_type='itervar', loss_type='rank', num_threads=None,
                 optimizer='sa', diversity_filter_ratio=None, log_interval=50):
        #####################
        cost_model = NNCostModel(task,
                                      feature_type=feature_type,
                                      loss_type=loss_type,
                                      num_threads=num_threads,
                                      log_interval=log_interval // 2)
        #####################
        if optimizer == 'sa':
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval)
        else:
            assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                          "a supported name string" \
                                                          "or a ModelOptimizer object."

        super(NNTuner, self).__init__(task, cost_model, optimizer,
                                       plan_size, diversity_filter_ratio)

    def tune(self, *args, **kwargs):  # pylint: disable=arguments-differ
        super(NNTuner, self).tune(*args, **kwargs)

        # manually close pool to avoid multiprocessing issues
        self.cost_model._close_pool()
