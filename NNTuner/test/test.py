import tvm
from tvm import autotvm

import tvm
import numpy
import timeit
from tvm import autotvm
import logging
import sys
import os
import numpy as np
def testwithNoneopt(str,ctx,matmul):
    # apply history best from log file
    autotvm.record.pick_best(str,str+".best")
    with autotvm.apply_history_best(str):
        with tvm.target.create("llvm"):
            s, arg_bufs = matmul(M,K,N, 'float32')
            func = tvm.build(s, arg_bufs)
    # check correctness
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
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
                                         'K = ' + str(K) + '\n'
                                         'N = ' + str(N) + '\n'
                                         'dtype = "float32"\n'
                                         'a = numpy.random.rand(M, K).astype(dtype)\n'
                                         'b = numpy.random.rand(K, N).astype(dtype)\n',
                                   stmt='answer = numpy.dot(a, b)',
                                   number=np_repeat)
    print("Numpy running time: %f" % (np_runing_time / np_repeat))

@autotvm.template
def matmul(M,K,N,dtype):
    # Algorithm
    k = tvm.reduce_axis((0, K), 'k')
    A = tvm.placeholder((M, K), name='A')
    B = tvm.placeholder((K, N), name='B')

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_x", M, num_outputs=3)
    cfg.define_split("tile_y", N, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=2)
    ##### define space end #####

    # We have to re-write the algorithm slightly.
    bn = cfg["tile_y"].size[-1]
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, y % bn], axis=k),
                    name = 'C')
    s = tvm.create_schedule(C.op)
    x, y = s[C].op.axis
    k, = s[C].op.reduce_axis

    # schedule according to config
    # Allocate write cache
    CC = s.cache_write(C, 'global')
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    s[C].reorder(xt, yt, xo, yo, xi, yi)
    xyt = s[C].fuse(xt, yt)
    # parallel
    s[C].parallel(xyt)
    xyo = s[C].fuse(xo, yo)
    s[C].unroll(xi)
    s[C].vectorize(yi)

    # Write cache is computed at xyo
    s[CC].compute_at(s[C], xyo)

    # New inner axes
    xc, yc = s[CC].op.axis

    k, = s[CC].op.reduce_axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    cfg.define_reorder("reorder", [xc, ki, yc], "all")
    cfg["reorder"].apply(s, CC, [xc, ki, yc])
    cfg.define_annotate('ann', [xc, ki, yc], policy='try_unroll_vec')
    cfg['ann'].apply(s, CC, [xc, ki, yc])


    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)

    return s, [A, B, C]


if __name__ == '__main__':
    # M = sys.argv[1]
    # L = sys.argv[2]
    # N = sys.argv[3]
    # M = int(M)
    # L = int(L)
    # N = int(N)
    M = 1024
    N = 1024
    K = 1024
    target = 'llvm'
    dtype = 'float32'
    ctx = tvm.context(target, 0)
    src = str(M) + "*" + str(K) + "*" + str(N)
    print(src)
    matmul = matmul
    space_len = 16
    early_stopping = 8
    task = autotvm.task.create(matmul,args=(M,K,N,dtype),target=target)
    print(task.config_space)
    testwithnumpy()
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
    XGBtuner.tune(n_trial=space_len,early_stopping=early_stopping, measure_option=measure_option, callbacks=[autotvm.callback.progress_bar(space_len),autotvm.callback.log_to_file('XGBtuner_matmul.log')])

    print("###############################")
    #testwithNoneopt('XGBtuner_matmul.log',ctx,matmul)
    testwithnumpy()
    print(XGBtuner.flops_max)
    print(XGBtuner.task)
    print(XGBtuner.xs)



