
import logging
import time
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import minmax_scale,scale,maxabs_scale,normalize,Binarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime


from sklearn.decomposition import PCA
logger = logging.getLogger('autotvm')
def count_time(func):
    def int_time(*args, **kwargs):
        start_time = datetime.datetime.now()  # 程序开始时间
        func(*args, **kwargs)
        over_time = datetime.datetime.now()   # 程序结束时间
        total_time = (over_time-start_time).total_seconds()
        print('程序共计%s秒' % total_time)
    return int_time

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
            ret[i] = np.sum(trial_ranks[:i] <= i) / (i+1)
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

def fit(x_train, y_train, xgb_params,plan_size,log_interval):
    #print("_____________fit______________")
    tic = time.time()

    valid_index = y_train > 1e-6

    index = np.random.permutation(len(x_train))

    #print("x_train shape:", x_train.shape)
    #print("y_train shape:", y_train.shape)
    #print("y train[:10]",y_train[:10])

    dtrain = xgb.DMatrix(x_train[index], y_train[index])
    sample_size = len(x_train)
    #print("sample size:",sample_size)

    bst = xgb.train(xgb_params, dtrain,
                         num_boost_round=8000,
                         callbacks=[custom_callback(
                             stopping_rounds=200,
                             metric='tr-a-recall@%d' % plan_size,
                             evals=[(dtrain, 'tr')],
                             maximize=True,
                             fevals=[xgb_average_recalln_curve_score(plan_size),],
                             verbose_eval=log_interval)])

    print("XGB train cost time:",time.time() - tic,"\tobs:",len(x_train),"\terror:\t",len(x_train) - np.sum(valid_index))
    return bst

def predict(bst, feas, output_margin=False):
    #print("___________predict___________")
    #print("feas shape:",feas.shape)
    dtest = xgb.DMatrix(feas)
    result = bst.predict(dtest, output_margin=output_margin)
    # print("predict result:",result)
    #print("predict result shape:",result.shape)
    return result
def load_data(datasrc,labelsrc):
    '''
    :param datasrc: 特征数据集 .txt     n*m
    :param labelsrc: 标签数据集 .txt    n*1
    :return:
    '''
    X = np.loadtxt(datasrc, delimiter=' ')
    Y = np.loadtxt(labelsrc, delimiter=' ')
    Y = np.array(Y).reshape((-1, 1))
    Y = np.array(Y)
    #y_max = np.max(Y)
    #Y = Y / max(y_max, 1e-8)
    print("X.shape:",X.shape)
    print("Y.shape:", Y.shape)
    #print(Y[:10])
    return X,Y
@count_time
def fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval):
    bst = fit(x_train, y_train, xgb_params, plan_size, log_interval)
    result = predict(bst, x_valiation)
    #print("result:\n", result[:10])
    #print("y_valiation:\n", y_valiation[:10])
    print("score:", mean_squared_error(result, y_valiation))

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


    X,Y= load_data(datasrc,labelsrc)

    print("situation1:original data")
    x_train, x_valiation, y_train, y_valiation = train_test_split(X, Y, test_size=0.1, random_state=42)
    fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval)

    print("situation2:nomalization(max-min) data")
    X1 = minmax_scale(X)
    Y1 = minmax_scale(Y)
    x_train, x_valiation, y_train, y_valiation = train_test_split(X1, Y1, test_size=0.1, random_state=42)
    fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval)

    print("situation3:nomalization(maxabs_scale) data")
    X2 = maxabs_scale(X)
    Y2 = maxabs_scale(Y)
    x_train, x_valiation, y_train, y_valiation = train_test_split(X2, Y2, test_size=0.1, random_state=42)
    fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval)

    print("situation4:pca")

    pca = PCA(n_components=9)
    X3 = pca.fit_transform(X)
    Y3 = Y
    #print("pca variance ratio", pca.explained_variance_ratio_)
    x_train, x_valiation, y_train, y_valiation = train_test_split(X3, Y3, test_size=0.1, random_state=42)
    fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval)


    print("situation5:pca-maxabs")
    pca = PCA(n_components=9)
    X4 = pca.fit_transform(X)
    X4 = maxabs_scale(X4)
    Y4 = maxabs_scale(Y)
    #print("pca variance ratio", pca.explained_variance_ratio_)
    x_train, x_valiation, y_train, y_valiation = train_test_split(X4, Y4, test_size=0.1, random_state=42)
    fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval)

    print("situation6:pca-maxmin only X")
    pca = PCA(n_components=9)
    X5 = pca.fit_transform(X)
    X5 = minmax_scale(X5)
    Y5 = Y
    #print("pca variance ratio", pca.explained_variance_ratio_)
    x_train, x_valiation, y_train, y_valiation = train_test_split(X5, Y5, test_size=0.1, random_state=42)
    fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval)


    print("situation7:pca-maxabs olny X")
    pca = PCA(n_components=9)
    X6 = pca.fit_transform(X)
    X6 = maxabs_scale(X6)
    Y6 = Y
    #print("pca variance ratio", pca.explained_variance_ratio_)
    x_train, x_valiation, y_train, y_valiation = train_test_split(X6, Y6, test_size=0.1, random_state=42)
    fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval)

    '''
        ValueError: Input contains NaN, infinity or a value too large for dtype('float32').

    print("situation9:pca-minmax both")
    pca = PCA(n_components=9)
    X8 = pca.fit_transform(X)
    X8 = minmax_scale(X8)
    Y8 = minmax_scale(Y)
    #print("pca variance ratio", pca.explained_variance_ratio_)
    x_train, x_valiation, y_train, y_valiation = train_test_split(X8, Y8, test_size=0.1, random_state=42)
    fit_and_evaluation(x_train, y_train, x_valiation, y_valiation, xgb_params, plan_size, log_interval)
    '''
'''
analysis:
    note:使用liner:gemma而不是默认的rank
                                        score           time            error
    situation1  origin(baseline)        0.0009096       13.403686       0
    situation2  max-min both            0.0010177       7.5144076       1
    situation3  max-abs both            0.0009096       12.203387       0
    situation4  pca                     0.0010287       6.8886067       0
    situation5  pca max-abs both        0.0010287       6.8756427       0
    situation6  pca max-min only X      0.0010287       6.7654283       0
    situation7  pca max-abs only X      0.0010287       6.8741359       0
    situation9  pca max-min both        may error
summary:
    pac + maxabs maybe a good choice! 
    
    在xgboost_cost_model fit()中：对X没有做任何处理，对Y进行了Y/max(Y)

'''



'''
Out:
situation1:original data
XGB train cost time: 13.403686761856079 	obs: 583 	error:	 0
score: 0.0009096166438441402
situation2:nomalization(max-min) data
XGB train cost time: 7.514407634735107 	obs: 583 	error:	 1
score: 0.0010177815519361315
situation3:nomalization(maxabs_scale) data
XGB train cost time: 12.203387975692749 	obs: 583 	error:	 0
score: 0.0009096166438441402
situation4:pca
XGB train cost time: 6.888606786727905 	obs: 583 	error:	 0
score: 0.0010287857382571318
situation5:pca-maxabs
XGB train cost time: 6.875642776489258 	obs: 583 	error:	 0
score: 0.0010287857382571318
situation6:pca-maxmin only X
XGB train cost time: 6.765428304672241 	obs: 583 	error:	 0
score: 0.0010287857382571318
situation7:pca-maxabs olny X
XGB train cost time: 6.874135971069336 	obs: 583 	error:	 0
score: 0.0010287857382571318
situation8:pca-maxabs both
XGB train cost time: 6.850676774978638 	obs: 583 	error:	 0
score: 0.0010287857382571318







'''


