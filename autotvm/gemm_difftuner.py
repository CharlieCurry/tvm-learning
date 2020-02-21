import logging
import numpy as np
import tvm
import sys
import math
import timeit
from tvm import autotvm

def matmul_v0(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    yo, yi = s[C].split(y, 8)
    xo, xi = s[C].split(x, 8)
    s[C].reorder(yo, xo, k, yi, xi)
    return s, [A, B, C]
# Matmul V1: List candidate values
@autotvm.template  # 1. use a decorator
def matmul_v1(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    # 2. get the config object
    cfg = autotvm.get_config()
    # 3. define search space
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16,32,64])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16,32,64])
    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg['tile_y'].val)
    xo, xi = s[C].split(x, cfg['tile_x'].val)
    s[C].reorder(yo, xo, k, yi, xi)
    #other optimization skills/tricks
    s[C].vectorize(yi)

    return s, [A, B, C]
@autotvm.template
def matmul_v2(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####
    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, k, yi, xi)
    return s, [A, B, C]

@autotvm.template
def matmul_v3(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    ##### define space begin #####
    cfg = autotvm.get_config()
    # use a filter that only accepts the split scheme whose inner most tile is less then 4
    cfg.define_split('tile_y', y, policy='factors', filter=lambda x: x.size[-1] <= 4)
    cfg.define_split('tile_y', x, policy='factors', filter=lambda x: x.size[-1] <= 4)
    ##### define space end #####
    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, k, yi, xi)
    return s, [A, B, C]

@autotvm.template
def matmul_v4(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    y, x = s[C].op.axis

    k = s[C].op.reduce_axis[0]

    cfg = autotvm.get_config()
    # define search space
    #tiling
    cfg.define_knob("tile_y", [1,2,4,8,16,32,64])
    cfg.define_knob("tile_x", [1,2,4,8,16,32,64])
    yo, yi = s[C].split(y, cfg['tile_y'].val)
    xo, xi = s[C].split(x, cfg['tile_x'].val)
    # cfg.define_split("tile_f", f, num_outputs=4)
    # cfg.define_split("tile_y", y, num_outputs=4)
    # cfg.define_split("tile_x", x, num_outputs=4)
    # cfg.define_split("tile_rc", rc, num_outputs=3)
    # cfg.define_split("tile_ry", ry, num_outputs=3)
    # cfg.define_split("tile_rx", rx, num_outputs=3)
    #reordering
    cfg.define_reorder("ordering",(yo, xo, k, yi, xi),policy="all")#interval_all

    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    #other optimization skills/tricks
    s[C].vectorize(yi)
    return s, [A, B, C]


def testwithNoneopt(str,ctx,matmul):
    # apply history best from log file
    autotvm.record.pick_best(str,str+".best")
    with autotvm.apply_history_best(str):
        with tvm.target.create("llvm"):
            s, arg_bufs = matmul(N, L, M, 'float32')
            func = tvm.build(s, arg_bufs)
    # check correctness
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)
    c_tvm = tvm.nd.empty(c_np.shape)
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-5)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    print(str)
    print('TVM: %f' % evaluator(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm).mean)

def testwithnumpy():
    np_repeat = 100
    np_runing_time = timeit.timeit(setup='import numpy\n'
                                         'M = ' + str(M) + '\n'
                                         'K = ' + str(L) + '\n'
                                         'N = ' + str(N) + '\n'
                                         'dtype = "float32"\n'
                                         'a = numpy.random.rand(M, K).astype(dtype)\n'
                                         'b = numpy.random.rand(K, N).astype(dtype)\n',
                                   stmt='answer = numpy.dot(a, b)',
                                   number=np_repeat)
    print("Numpy running time: %f" % (np_runing_time / np_repeat))

if __name__ == '__main__':
    # M = sys.argv[1]
    # L = sys.argv[2]
    # N = sys.argv[3]
    # M = int(M)
    # L = int(L)
    # N = int(N)
    M = 512
    N = 512
    L = 224
    target = 'llvm'
    dtype = 'float32'
    ctx = tvm.context(target, 0)
    src = str(M) + "*" + str(L) + "*" + str(N)
    print(src)
    matmul = matmul_v4
    space_len = 100

    task = autotvm.task.create(matmul,args=(N,L,M,dtype),target=target)
    print(task.config_space)
    # logging config (for printing tuning log to the screen)
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)
    # logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    # There are two steps for measuring a config: build and run.
    # By default, we use all CPU cores to compile program. Then measure them sequentially.
    # We measure 5 times and take average to reduce variance.
    measure_option = autotvm.measure_option(builder='local',runner=autotvm.LocalRunner(number=5))
    # Begin tuning with RandomTuner, log records to file `matmul.log`
    # You can use alternatives like XGBTuner.
    print("XGBoost:")
    XGBtuner = autotvm.tuner.XGBTuner(task)
    XGBtuner.tune(n_trial=space_len, measure_option=measure_option, callbacks=[autotvm.callback.progress_bar(space_len),autotvm.callback.log_to_file('XGBtuner_matmul.log')])
    print("RandomSearch:")
    Rtuner = autotvm.tuner.RandomTuner(task)
    Rtuner.tune(n_trial=space_len,measure_option=measure_option,callbacks=[autotvm.callback.progress_bar(space_len),autotvm.callback.log_to_file('Rtuner_matmul.log')])
    print("GA:")
    Gatuner = autotvm.tuner.GATuner(task)
    Gatuner.tune(n_trial=space_len,measure_option=measure_option,callbacks=[autotvm.callback.progress_bar(space_len),autotvm.callback.log_to_file('Gatuner_matmul.log')])
    print("GridSearch")
    Grtuner = autotvm.tuner.GridSearchTuner(task)
    Grtuner.tune(n_trial=space_len,measure_option=measure_option,callbacks=[autotvm.callback.progress_bar(space_len),autotvm.callback.log_to_file('Gruner_matmul.log')])



    print("###############################")
    testwithNoneopt('Rtuner_matmul.log',ctx,matmul)
    testwithNoneopt('Gatuner_matmul.log',ctx,matmul)
    testwithNoneopt('XGBtuner_matmul.log',ctx,matmul)
    testwithNoneopt('Gruner_matmul.log',ctx,matmul)
    testwithnumpy()
