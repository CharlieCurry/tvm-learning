import tvm
import numpy
import timeit
from tvm import autotvm
import logging
import sys
import os
# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(target, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)

np_repeat = 1000
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

answer = numpy.dot(a.asnumpy(), b.asnumpy())

@autotvm.template
def matmul():
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

task = autotvm.task.create(matmul, args=[], target=target)

measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

# begin tuning, log records to file `matmul.log`
tuner = autotvm.tuner.XGBTuner(task)
print(task.config_space)
n_trial = 2000
early_stopping = 800
if os.path.exists('matmul.log.tmp'):
    os.remove('matmul.log.tmp')
tuner.tune(n_trial=n_trial,
           early_stopping=early_stopping,
           measure_option=measure_option,
           callbacks=[autotvm.callback.progress_bar(n_trial),
                       autotvm.callback.log_to_file('matmul.log.tmp')])
# pick best records to a cache file
autotvm.record.pick_best('matmul.log.tmp', 'matmul.log')

with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create('llvm -mcpu=core-avx2'):
        s, arg_buf = matmul()
        func = tvm.build(s, arg_buf)
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)


func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
# print(func.get_source("asm"))

evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
print('TVM: %f' % evaluator(a, b, c).mean)
# print(tvm.lower(s, arg_buf, simple_mode=True))