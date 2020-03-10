import logging
import numpy as np
import tvm
import random
import sys
import timeit
from tvm import relay
import datetime
from tvm import autotvm

'''
相比v:在tile factor定义采取了不同的方式，比之前的空间更小
'''
def numpyBaseline(M,K,N):
    np_repeat = 100
    np_runing_time = timeit.timeit(setup='import numpy\n'
                                         'M = ' + str(M) + '\n'
                                        'K = ' + str(K) + '\n'
                                        'N = ' + str(N) + '\n'
                                        'dtype = "float32"\n'
                                        'a = numpy.random.rand(M, K).astype(dtype)\n'
                                       'b = numpy.random.rand(K, N).astype(dtype)\n',
                                   stmt='answer = numpy.dot(a, b)',
                                   number=np_repeat)

    numpytime = (np_runing_time / np_repeat)
    print("Numpy running time: %f" % numpytime)
    return numpytime

def buildandevaluation(s,A,B,C,a,b,c,ctx,c_np):
    with relay.build_config(opt_level=3):
        func = tvm.build(s, [A, B, C], target=target, name='gemm')
    assert func
    func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=20)
    result =  evaluator(a, b, c).mean
    print('evaluation time: %.10f' %result)
    #print(tvm.lower(s, [A, B, C], simple_mode=True))
    return result

def apply_history_best(str,ctx,method):
    # apply history best from log file
    autotvm.record.pick_best(str,str+".best")
    with tvm.target.create('llvm'):
        with autotvm.apply_history_best(str):
            s, arg_bufs = method(N, K, M, 'float32')
            func = tvm.build(s, arg_bufs)
    # check correctness
    a_np = np.random.uniform(size=(N, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, M)).astype(np.float32)
    c_np = a_np.dot(b_np)
    c_tvm = tvm.nd.empty(c_np.shape)
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-5)
    #time evaluation
    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    tvm_time = evaluator(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm).mean
    print("speedup= %.10f" % (numpyBaseline(M, K, N) / tvm_time))

def Gemm_tv2_reorder2_3_vec1_para1(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts=None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo,yo,ko,xi,ki,yi)
    #实验表明（无论从可行性和性能上都）应该针对最内层
    s[C].vectorize(yi)
    #实验表明，应当针对最外层的循环轴进行parallel
    s[C].parallel(xo)
    return s

def Gemm_tv2_reorder2_3_vec1_para1_config_define(N, K, M, dtype):
    A = tvm.placeholder((N, K), name='A', dtype=dtype)
    B = tvm.placeholder((K, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')

    s = tvm.create_schedule(C.op)
    k = s[C].op.reduce_axis[0]
    y, x = s[C].op.axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x , num_outputs=2)
    cfg.define_split("tile_y", y , num_outputs=2)
    cfg.define_split("tile_k", k , num_outputs=2)
    '''
    >>> # use custom candidates
    >>> cfg.define_split('tile_x', x, policy='candidate', candidate=[[1, 4, 4], [4, 1, 4]])'''
    #>>> # use a filter that only accepts the split scheme whose inner most tile is less then 64
    # cfg.define_split('tile_x', x, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_y', y, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_k', k, policy='factors', filter=lambda x: x.size[-1] <= 64)

    # cfg.define_knob("tile_x", [1, 4, 8, 16, 32, 64])
    # cfg.define_knob("tile_y", [1, 4, 8, 16, 32, 64])
    # cfg.define_knob("tile_k", [1, 4, 8, 16, 32, 64])
    # xo, xi = s[C].split(x, cfg['tile_x'].val)
    # yo, yi = s[C].split(y, cfg['tile_y'].val)
    # ko, ki = s[C].split(k, cfg['tile_k'].val)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    ko, ki = cfg["tile_k"].apply(s, C, k)
    s[C].reorder(xo,yo,ko,xi,ki,yi)
    s[C].vectorize(yi)
    s[C].parallel(xo)
    return s, [A, B, C]

def Gemm_tv2_reorder2_3_vec1_para1_unrollv1(C):
    s = tvm.create_schedule(C.op)
    bn = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    s[C].reorder(xo,yo,ko,xi,ki,yi)
    s[C].vectorize(yi)
    s[C].parallel(xo)
    s[C].unroll(ki)
    return s

def Gemm_tv2_reorder2_3_vec1_para1_unrollv1_best_config(C):
    s = tvm.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 8, 16)
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=64,nparts =None)
    s[C].reorder(xo,yo,ko,xi,ki,yi)
    s[C].vectorize(yi)
    s[C].parallel(xo)
    s[C].unroll(ki)
    return s

def Gemm_tv2_reorder2_3_vec1_para1_unrollv1_config_define(N, K, M, dtype):
    A = tvm.placeholder((N, K), name='A', dtype=dtype)
    B = tvm.placeholder((K, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')

    s = tvm.create_schedule(C.op)
    k = s[C].op.reduce_axis[0]
    y, x = s[C].op.axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_k", k, num_outputs=2)
    # cfg.define_split('tile_x', x, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_y', y, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_k', k, policy='factors', filter=lambda x: x.size[-1] <= 64)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    ko, ki = cfg["tile_k"].apply(s, C, k)
    # cfg.define_knob("tile_x", [1, 4, 8, 16, 32, 64])
    # cfg.define_knob("tile_y", [1, 4, 8, 16, 32, 64])
    # cfg.define_knob("tile_k", [1, 4, 8, 16, 32, 64])
    # xo, xi = s[C].split(x, cfg['tile_x'].val)
    # yo, yi = s[C].split(y, cfg['tile_y'].val)
    # ko, ki = s[C].split(k, cfg['tile_k'].val)

    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    s[C].parallel(xo)
    s[C].unroll(ki)
    return s, [A, B, C]

def Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1(C):
    bn = 32
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N), lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, yo, ko, xi,ki,yi)
    #实验表明（无论从可行性和性能上都）应该针对最内层
    s[C].vectorize(yi)
    #实验表明，应当针对最外层的循环轴进行parallel
    s[C].parallel(xo)
    #实验表明，unroll用于倒数第二层效果好
    s[C].unroll(ki)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    s[packedB].unroll(y)
    return s

def Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1_config_define(N, K, M, dtype):
    A = tvm.placeholder((N, K), name='A', dtype=dtype)
    B = tvm.placeholder((K, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, K), name='k')

    bn = 32
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N), lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    k = s[C].op.reduce_axis[0]
    y, x = s[C].op.axis

    cfg = autotvm.get_config()
    #cfg.define_knob("bn",candidate=[1,2,4,8,16,32,64])
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_k", k, num_outputs=2)
    # cfg.define_split('tile_x', x, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_y', y, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_k', k, policy='factors', filter=lambda x: x.size[-1] <= 64)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    ko, ki = cfg["tile_k"].apply(s, C, k)
    # cfg.define_knob("tile_x", [1, 4, 8, 16, 32, 64])
    # cfg.define_knob("tile_y", [1, 4, 8, 16, 32, 64])
    # cfg.define_knob("tile_k", [1, 4, 8, 16, 32, 64])
    # xo, xi = s[C].split(x, cfg['tile_x'].val)
    # yo, yi = s[C].split(y, cfg['tile_y'].val)
    # ko, ki = s[C].split(k, cfg['tile_k'].val)

    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    s[C].parallel(xo)
    s[C].unroll(ki)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    s[packedB].unroll(y)
    return s, [A, B, C]

def Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1_writecachev1(C):
    bn = 32
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N), lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    # Allocate write cache
    CC = s.cache_write(C, 'global')
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    # Write cache is computed at yo
    s[CC].compute_at(s[C], yo)
    ko, ki = s[CC].split(s[CC].op.reduce_axis[0], factor=32,nparts =None)

    # New inner axes
    xc, yc = s[CC].op.axis
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].vectorize(yc)
    #s[CC].parallel(ko)不能加???
    s[CC].unroll(ki)

    s[C].parallel(xo)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    s[packedB].unroll(y)
    return s

def Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1_writecachev1_config_define(N, K, M, dtype):
    A = tvm.placeholder((N, K), name='A', dtype=dtype)
    B = tvm.placeholder((K, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, K), name='k')

    bn = 32
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N), lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    k = s[C].op.reduce_axis[0]
    C = s.cache_write(C, 'global')
    y, x = s[C].op.axis

    cfg = autotvm.get_config()
    #cfg.define_knob("bn",candidate=[1,2,4,8,16,32,64])
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_k", k, num_outputs=2)
    # cfg.define_split('tile_x', x, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_y', y, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_k', k, policy='factors', filter=lambda x: x.size[-1] <= 64)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    ko, ki = cfg["tile_k"].apply(s, C, k)
    # cfg.define_knob("tile_x", [1, 4, 8, 16, 32, 64])
    # cfg.define_knob("tile_y", [1, 4, 8, 16, 32, 64])
    # cfg.define_knob("tile_k", [1, 4, 8, 16, 32, 64])
    # xo, xi = s[C].split(x, cfg['tile_x'].val)
    # yo, yi = s[C].split(y, cfg['tile_y'].val)
    # ko, ki = s[C].split(k, cfg['tile_k'].val)

    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    s[C].parallel(xo)
    s[C].unroll(ki)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    s[packedB].unroll(y)
    return s, [A, B, C]

def find_bestSchedule(A, B, C, a, b, c, ctx, c_np):
    res = {}
    print("########################Schedule candidate########################")
    print("parallel")
    s = Gemm_tv2_reorder2_3_vec1_para1(C)
    res['parallel'] = buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("unroll")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1(C)
    res['unroll'] = buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("packing")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1(C)
    res['packing'] = buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("write cache")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1_writecachev1(C)
    res['write cache'] = buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("#######################find best schedule#########################")
    key_min = min(res.keys(), key=(lambda k: res[k]))

    print('Minimum Value: %.10f'%res[key_min])
    print(key_min)
    print("speedup= %.10f"%(numpyBaseline(M, K, N) / res[key_min]))
    print("##################################################################")
    return key_min

def useTuner(tuner,space_len,task,measure_option):
    print(task.config_space)
    if tuner=="all":
        print("XGBoost:")
        XGBtuner = autotvm.tuner.XGBTuner(task)
        XGBtuner.tune(n_trial=space_len, early_stopping=space_len/5,measure_option=measure_option, callbacks=[autotvm.callback.progress_bar(space_len),
                                                                               autotvm.callback.log_to_file(
                                                                        'XGBTuner_matmul.log')])
        print("RandomSearch:")
        Rtuner = autotvm.tuner.RandomTuner(task)
        Rtuner.tune(n_trial=space_len,  early_stopping=space_len/5,measure_option=measure_option,
                callbacks=[autotvm.callback.progress_bar(space_len), autotvm.callback.log_to_file('RDTuner_matmul.log')])
        print("GA:")
        Gatuner = autotvm.tuner.GATuner(task)
        Gatuner.tune(n_trial=space_len,  early_stopping=space_len/5,measure_option=measure_option, callbacks=[autotvm.callback.progress_bar(space_len),
                                                                              autotvm.callback.log_to_file(
                                                                                  'GATuner_matmul.log')])
        print("GridSearch:")
        Grtuner = autotvm.tuner.GridSearchTuner(task)
        Grtuner.tune(n_trial=space_len,  early_stopping=space_len/5,measure_option=measure_option, callbacks=[autotvm.callback.progress_bar(space_len),
                                                                              autotvm.callback.log_to_file(
                                                                                  'GRTuner_matmul.log')])
        return XGBtuner,Rtuner,Gatuner,Grtuner

    if tuner=="xgb":
        print("XGBoost:")
        XGBtuner = autotvm.tuner.XGBTuner(task)
        XGBtuner.tune(n_trial=space_len, measure_option=measure_option,
                      callbacks=[autotvm.callback.progress_bar(space_len),
                                 autotvm.callback.log_to_file(
                                     'XGBTuner_matmul.log')])
        return XGBtuner
    if tuner=="rs":
        print("RandomSearch:")
        Rtuner = autotvm.tuner.RandomTuner(task)
        Rtuner.tune(n_trial=space_len, measure_option=measure_option,
                    callbacks=[autotvm.callback.progress_bar(space_len),
                               autotvm.callback.log_to_file('RDTuner_matmul.log')])
        return Rtuner
    if tuner=="ga":
        print("GA:")
        Gatuner = autotvm.tuner.GATuner(task)
        Gatuner.tune(n_trial=space_len, measure_option=measure_option,
                     callbacks=[autotvm.callback.progress_bar(space_len),
                                autotvm.callback.log_to_file(
                                    'GATuner_matmul.log')])
        return Gatuner
    if tuner=="gs":
        print("GridSearch:")
        Grtuner = autotvm.tuner.GridSearchTuner(task)
        Grtuner.tune(n_trial=space_len, measure_option=measure_option,
                     callbacks=[autotvm.callback.progress_bar(space_len),
                                autotvm.callback.log_to_file(
                                    'GRTuner_matmul.log')])
        return Grtuner
    if tuner=="none":
        print("do nothing.")
        return None


def xgb_more_Info(myTuner):
    print("fea type:",myTuner.cost_model.fea_type)
    feature_cache = myTuner.cost_model.feature_cache
    print("feature_cache type:",type(feature_cache))
    feature_cache_context = feature_cache.get(myTuner.cost_model.fea_type)
    print("feature cache context type:",type(feature_cache_context))
    print("feature_cache_context len:",len(feature_cache_context))
    print("feature_cache_context keys:",feature_cache_context.keys())

    #print("feature_cache_context values",feature_cache_context.values())
    #np.save('feature_cache_context.npy', feature_cache_context)
    print("task config space map:",myTuner.task.config_space.space_map)
    print("trails",myTuner.trials)
    print("plan size:",myTuner.plan_size)
    print("diversity_filter_ratio:",myTuner.diversity_filter_ratio)
    print("space",myTuner.space)
    print("task:",myTuner.task)
    print("task config space:",myTuner.task.config_space)
    print("flops max:",myTuner.flops_max)
    print("best_iter",myTuner.best_iter)
    print("tuner param:",myTuner.param)
    print("tuner dims:",myTuner.dims)
    print("xs:",myTuner.xs)
    print("ys:",myTuner.ys)
    print("x_train:",myTuner.cost_model.x_train)
    print("x_train shape:", myTuner.cost_model.x_train.shape)
    print("y_train:",myTuner.cost_model.y_train)
    print("y_train shape:", myTuner.cost_model.y_train.shape)
    print("feas:",myTuner.cost_model.feas)
    print("feas shape:", myTuner.cost_model.feas.shape)
    # np.savetxt("feas.txt", myTuner.cost_model.feas, fmt='%s', delimiter=' ')
    # np.savetxt("x_train.txt", myTuner.cost_model.x_train, fmt='%s', delimiter=' ')
    # np.savetxt("y_train.txt", myTuner.cost_model.y_train, fmt='%s', delimiter=' ')
    # np.savetxt("xs.txt", myTuner.xs, fmt='%s', delimiter=' ')
    # np.savetxt("ys.txt", myTuner.ys, fmt='%s', delimiter=' ')
    print("predict count:",myTuner.cost_model.predict_count)
    print("space get(index = 64)",myTuner.space.get(64))
    print("myTuner rets:",myTuner.rets)
    print("predict count:",myTuner.cost_model.predict_count)
    print("fit count:",myTuner.cost_model.fit_count)
    print("find maximums count:",myTuner.model_optimizer.find_maximums_count)


def grid_more_Info(myTuner):
    '''
    在gridsearch_tuner中添加GRrets,GRindexs
    '''
    print("GRrets:",myTuner.GRrets)
    print("GRindexs:",myTuner.GRindexs)
    print("config_space get(index = 64)",myTuner.task.config_space.get(64))

def count_time(func):
    def int_time(*args, **kwargs):
        start_time = datetime.datetime.now()  # 程序开始时间
        func(*args, **kwargs)
        over_time = datetime.datetime.now()   # 程序结束时间
        total_time = (over_time-start_time).total_seconds()
        print('程序共计%s秒' % total_time)
    return int_time


if __name__ == '__main__':
    # M = sys.argv[1]
    # K = sys.argv[2]
    # N = sys.argv[3]
    # M = int(M)
    # K = int(K)
    # N = int(N)
    M = 128
    N = 256
    K = 256
    print(str(M)+"*"+str(K)+"*"+str(N))
    random.seed(30)
    target = 'llvm'
    dtype = 'float32'
    ctx = tvm.context(target, 0)
    k = tvm.reduce_axis((0, K), 'k')
    A = tvm.placeholder((M, K), name='A')
    B = tvm.placeholder((K, N), name='B')
    C = tvm.compute((M, N),lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),name='C')

    a_np = np.random.rand(M,K).astype(dtype)
    b_np = np.random.rand(K,N).astype(dtype)
    c_np = np.zeros((M,N)).astype(dtype)
    c_np = a_np.dot(b_np)

    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(c_np, ctx)
    s = tvm.create_schedule(C.op)
    dict_autoschedule = {'parallel':Gemm_tv2_reorder2_3_vec1_para1_config_define,
                         'unroll':Gemm_tv2_reorder2_3_vec1_para1_unrollv1_config_define,
                         'packing':Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1_config_define,
                         'write cache':Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1_writecachev1_config_define}


    key = find_bestSchedule(A, B, C, a, b, c, ctx, c_np)
    print("best schedule method: ",dict_autoschedule[key])

    task = autotvm.task.create(dict_autoschedule[key], args=(N, K, M, dtype), target=target)
    measure_option = autotvm.measure_option(builder='local', runner=autotvm.LocalRunner(number=5))
    myTuner = useTuner("xgb",148,task,measure_option)
    #apply_history_best('XGBtuner_matmul.log',ctx,Gemm_tv2_reorder2_3_vec1_para1_config_define)
    #testwithNoneopt('XGBtuner_matmul.log', ctx, Gemm_tv2_reorder2_3_vec1_para1_config_define)
    # apply history best from log file
    str = 'XGBTuner_matmul.log'
    autotvm.record.pick_best(str, str + ".best")
    # str = 'GATuner_matmul.log'
    # autotvm.record.pick_best(str, str + ".best")
    # str = 'RDTuner_matmul.log'
    # autotvm.record.pick_best(str, str + ".best")
    # str = 'GRTuner_matmul.log'
    #autotvm.record.pick_best(str, str + ".best")
##这里定义的schedule是具体的,没有暴露出可调节的参数,感觉这里任然是手工调节，并且在面对不同的问题上有较大的不确定性

    #buildandevaluation(Gemm_tv2_reorder2_3_vec1_para1_unrollv1_best_config(C), A, B, C, a, b, c, ctx, c_np)
    #xgb_more_Info(myTuner)
    xgb_more_Info(myTuner)





