import warnings
warnings.filterwarnings("ignore")
import logging
import time
import numpy as np
import xgboost as xgb
from TSSModel import TSSModel
from sklearn.preprocessing import minmax_scale, scale, maxabs_scale, normalize, Binarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA

logger = logging.getLogger('autotvm')


def recall_curve(trial_ranks, top=None):
    """
    if top is None, f(n) = sum([I(rank[i] < n) for i < n]) / n
    if top is K,    f(n) = sum([I(rank[i] < K) for i < n]) / K

    Parameters
    ----------
    trial_ranks: Array of int
        the rank of i th trial in labels
    top: int or None
        top-n recall

    Returns
    -------
    curve: Array of float
        function values
    """
    if not isinstance(trial_ranks, np.ndarray):
        trial_ranks = np.array(trial_ranks)

    ret = np.zeros(len(trial_ranks))
    if top is None:
        for i in range(len(trial_ranks)):
            ret[i] = np.sum(trial_ranks[:i] <= i) / (i + 1)
    else:
        for i in range(len(trial_ranks)):
            ret[i] = 1.0 * np.sum(trial_ranks[:i] < top) / top
    return ret


def get_rank(values):
    """get rank of items

    Parameters
    ----------
    values: Array

    Returns
    -------
    ranks: Array of int
        the rank of this item in the input (the largest value ranks first)
    """
    tmp = np.argsort(-values)
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(tmp))
    return ranks


def xgb_average_recalln_curve_score(N):
    """evaluate average recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@%d" % N, np.sum(curve[:N]) / N

    return feval


def custom_callback(stopping_rounds, metric, fevals, evals=(), log_file=None,
                    maximize=False, verbose_eval=True):
    """callback function for xgboost to support multiple custom evaluation functions"""
    from xgboost.core import EarlyStopException
    from xgboost.callback import _fmt_metric
    from xgboost.training import aggcv

    state = {}
    metric_shortname = metric.split("-")[1]

    def init(env):
        """internal function"""
        bst = env.model

        state['maximize_score'] = maximize
        state['best_iteration'] = 0
        if maximize:
            state['best_score'] = float('-inf')
        else:
            state['best_score'] = float('inf')

        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        if not state:
            init(env)

        bst = env.model
        i = env.iteration
        cvfolds = env.cvfolds

        res_dict = {}

        ##### evaluation #####
        if cvfolds is not None:
            for feval in fevals:
                tmp = aggcv([f.eval(i, feval) for f in cvfolds])
                for k, mean, std in tmp:
                    res_dict[k] = [mean, std]
        else:
            for feval in fevals:
                bst_eval = bst.eval_set(evals, i, feval)
                res = [x.split(':') for x in bst_eval.split()]
                for kv in res[1:]:
                    res_dict[kv[0]] = [float(kv[1])]

        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        ##### print eval result #####
        infos = ["XGB iter: %3d" % i]
        for item in eval_res:
            if 'null' in item[0]:
                continue
            infos.append("%s: %.6f" % (item[0], item[1]))

        if not isinstance(verbose_eval, bool) and verbose_eval and i % verbose_eval == 0:
            logger.debug("\t".join(infos))
        if log_file:
            with open(log_file, "a") as fout:
                fout.write("\t".join(infos) + '\n')

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == metric:
                score = item[1]
                break
        assert score is not None

        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        if (maximize_score and score > best_score) or \
                (not maximize_score and score < best_score):
            msg = '[%d] %s' % (
                env.iteration,
                '\t'.join([_fmt_metric(x) for x in eval_res]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state['best_msg']
            if verbose_eval and env.rank == 0:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback


def fit(x_train, y_train, xgb_params, plan_size, log_interval):
    # print("_____________fit______________")
    tic = time.time()

    valid_index = y_train > 1e-6

    index = np.random.permutation(len(x_train))

    # print("x_train shape:", x_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("y train[:10]",y_train[:10])

    dtrain = xgb.DMatrix(x_train[index], y_train[index])
    sample_size = len(x_train)
    # print("sample size:",sample_size)

    bst = xgb.train(xgb_params, dtrain,
                    num_boost_round=8000,
                    callbacks=[custom_callback(
                        stopping_rounds=20,
                        metric='tr-a-recall@%d' % plan_size,
                        evals=[(dtrain, 'tr')],
                        maximize=True,
                        fevals=[xgb_average_recalln_curve_score(plan_size), ],
                        verbose_eval=log_interval)])

    print("XGB train cost time:", time.time() - tic, "\tobs:", len(x_train), "\terror:\t",
          len(x_train) - np.sum(valid_index))
    return bst


def predict(bst, feas, output_margin=False):
    # print("___________predict___________")
    # print("feas shape:",feas.shape)
    dtest = xgb.DMatrix(feas)
    result = bst.predict(dtest, output_margin=output_margin)
    # print("predict result:",result)
    # print("predict result shape:",result.shape)
    return result


def load_data(datasrc, labelsrc):
    '''
    :param datasrc: 特征数据集 .txt     n*m
    :param labelsrc: 标签数据集 .txt    n*1
    :return:
    '''
    X = np.loadtxt(datasrc, delimiter=' ')
    Y = np.loadtxt(labelsrc, delimiter=' ')
    Y = np.array(Y).reshape((-1, 1))
    Y = np.array(Y)
    # y_max = np.max(Y)
    # Y = Y / max(y_max, 1e-8)
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    # print(Y[:10])
    return X, Y


def fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval):
    bst = fit(x_train, y_train, xgb_params, plan_size, log_interval)
    result = predict(bst, x_valiation)
    return result
    # print("result:\n", result[:10])
    # print("y_valiation:\n", y_valiation[:10])
    # print("mean_squared_error:", mean_squared_error(result, y_valiation))


if __name__ == '__main__':

    '''
    测试是否经PCA和标准化后xgboost模型的性能表现    
    '''
    datasrc = "train/feas1.txt"
    labelsrc = "train/y_train.txt"
    xgb_params = {
        'max_depth': 3,
        'gamma': 0.0001,
        'min_child_weight': 1,
        'subsample': 1.0,
        'eta': 0.3,
        'lambda': 1.00,
        'alpha': 0,
        'objective': 'reg:gamma',
    }
    plan_size = 64
    log_interval = 25



    input_size = 9
    tss = TSSModel(input_size=input_size)
    tss.load_data('train/feas1.txt', 'train/y_train.txt')




    print("xgb:situation1:split")
    tss.split(tss.X, tss.Y)
    y_prediction = fit_and_evaluation(tss.x_train, tss.y_train, tss.X, tss.Y, xgb_params, plan_size, log_interval)
    tss.envaluation(y_prediction, tss.Y)

    print("nntss:situation1")
    tss.pca_minmax_split(tss.X, tss.Y,pca_components=input_size)
    tss.TSS_NN_Model_fit(tss.x_train, tss.y_train, tss.x_valiation, tss.y_valiation)
    #print("predict(x_valiation,y_valiation)")
    y_prediction = tss.predict(tss.x_valiation, tss.y_valiation)
    tss.envaluation(tss.y_prediction, tss.y_valiation)

    #
    # print("xgb:situation2:maxmin split")
    # tss.minmax_split(tss.X,tss.Y)
    # fit_and_evaluation(tss.x_train, tss.y_train, tss.x_valiation, tss.y_valiation, xgb_params, plan_size, log_interval)
    #
    # print("xgb:situation3:maxabs split")
    # tss.minmax_split(tss.X, tss.Y)
    # fit_and_evaluation(tss.x_train, tss.y_train, tss.x_valiation, tss.y_valiation, xgb_params, plan_size, log_interval)
    #
    # print("xgb:situation4:nomalize split")
    # tss.minmax_split(tss.X, tss.Y)
    # fit_and_evaluation(tss.x_train, tss.y_train, tss.x_valiation, tss.y_valiation, xgb_params, plan_size, log_interval)
    #
    #
    # print("xgb:situation5:scale split")
    # tss.minmax_split(tss.X, tss.Y)
    # fit_and_evaluation(tss.x_train, tss.y_train, tss.x_valiation, tss.y_valiation, xgb_params, plan_size, log_interval)
    #
