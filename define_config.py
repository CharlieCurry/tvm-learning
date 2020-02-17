import logging
import numpy as np
import tvm
import sys
import math
import timeit
from tvm import autotvm
#
#本教程主要展示如何定义一个比较合适的schedule congratulations以供autotvm调优
#Use better space definition API
# In the previous template, we manually list all possible values for a knob.
# This is the lowest level API to define the space. However,
# we also provide another set of API to make the space definition easier and smarter.
# It is recommended to use this set of high level API.
# In the following example, we use [ConfigSpace.define_split] to define a split knob.
# It will enumerate all the possible ways to split an axis and construct the space.
# We also have [ConfigSpace.define_reorder] for reorder knob and [ConfigSpace.define_annotate] for annotation like unroll,
# vectorization, thread binding.
#  When the high level API cannot meet your requirement, you can always fall back to use low level API.
#
def matmul_v1(N, L, M, dtype):
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
    cfg.define_annotate("unroll",policy="try_unroll")
    #policy (str) – name of policy If is ‘unroll’, unroll the axes.
    # If is ‘try_unroll’, try to unroll the axes.
    # If is ‘try_unroll_vec’, try to unroll or vectorize the axes.
    # If is ‘bind_gpu’, bind the first few axes to gpu threads.
    # If is ‘locate_cache’, choose n axes to attach shared/local cache.
    cfg.define_annotate("vec",policy="try_unroll_vec")
    cfg.define_annotate("cache",policy="locate_cache")
    #policy (str) – name of policy If is ‘identity’, do an identity permutation.
    # If is ‘all’, try all permutations.
    # If is ‘interval_all’, try all permutations of an interval of axes.
    # If is ‘candidate’, try listed candidate.
    # If is ‘interleave’, interleave chains of spatial axes and chains of reduction axes.
    cfg.define_reorder("reorder",policy="interval_all")
    ##### define space end #####
    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, k, yi, xi)
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
    target = 'llvm -mcpu=core-avx2'
    dtype = 'float32'
    ctx = tvm.context(target, 0)
    src = str(M) + "*" + str(L) + "*" + str(N)
    print(src)
    matmul = matmul_v1
    space_len = 100

    task = autotvm.task.create(matmul, args=(N,L,M,dtype), target=target)
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