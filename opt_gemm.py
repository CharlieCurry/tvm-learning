import logging
import numpy as np
import tvm
import random
import sys
import math
import timeit
from tvm import relay
from tvm import autotvm

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
    print("Numpy running time: %f" % (np_runing_time / np_repeat))

def buildandevaluation(s,A,B,C,a,b,c,ctx,c_np):
    with relay.build_config(opt_level=3):
        func = tvm.build(s, [A, B, C], target=target, name='gemm')
    assert func
    func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    print('time: %f' % evaluator(a, b, c).mean)
    #print(tvm.lower(s, [A, B, C], simple_mode=True))

def Gemm_v0(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    print("Tiling")
    return s

def Gemm_v1(C):
    # Tiling、reordering
    s = tvm.create_schedule(C.op)
    bn = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    s[C].reorder(xo, yo, ko, ki, xi, yi)
    print("Tiling、reordering")
    return s

def Gemm_v2(C):
    # Tiling、reordering、Vectorization
    bn = 32
    s = tvm.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    s[C].reorder(xo, yo, ko, ki, xi, yi)
    # Vectorization
    s[C].vectorize(yi)
    print("Tiling、reordering、Vectorization")
    return s

def Gemm_v3(C):
    # Tiling、reordering、Vectorization、Loop Permutation
    s = tvm.create_schedule(C.op)
    bn = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    # re-ordering    注意这里和v2不同
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    print("Tiling、reordering、Vectorization、Loop Permutation(reorder)")
    return s

def Gemm_v4(C):
    # Tiling(blocking)、reordering、Vectorization、Loop Permutation、Array Packing
    s = tvm.create_schedule(C.op)# Tiling、reordering、Vectorization、Loop Permutation、Array Packing
    bn = 32
    k, = s[C].op.reduce_axis
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k),
                    name='C')
    s = tvm.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    ko, ki = s[C].split(k, factor=4)
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    print("Tiling、reordering、Vectorization、Loop Permutation、Array Packing")
    return s

def Gemm_v5(C):
    # Tiling、reordering、Vectorization、Loop Permutation、Array Packing、Write cache for blocks
    bn = 32
    s = tvm.create_schedule(C.op)
    k, = s[C].op.reduce_axis
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k),
                    name='C')
    s = tvm.create_schedule(C.op)
    # Allocate write cache
    CC = s.cache_write(C, 'global')
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    # Write cache is computed at yo
    s[CC].compute_at(s[C], yo)
    # New inner axes
    xc, yc = s[CC].op.axis
    k, = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    print("Tiling、reordering、Vectorization、Loop Permutation、Array Packing、Write cache for blocks")
    return s

def Gemm_v6(C):
    # Tiling、reordering、Vectorization、Loop Permutation、Array Packing、Write cache for blocks、Parallel
    bn = 32
    s = tvm.create_schedule(C.op)
    k, = s[C].op.reduce_axis
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k),
                    name='C')
    s = tvm.create_schedule(C.op)
    # Futhermore, we can also utilize multi-core processors to do the thread-level parallelization.
    CC = s.cache_write(C, 'global')
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    s[CC].compute_at(s[C], yo)
    xc, yc = s[CC].op.axis
    k, = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)
    # parallel
    s[C].parallel(xo)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    print("Tiling、reordering、Vectorization、Loop Permutation、Array Packing、Write cache for blocks、Parallel")
    return s


if __name__ == '__main__':
    #M = sys.argv[1]
    #K = sys.argv[2]
    #N = sys.argv[3]
    #M = int(M)
    #K = int(K)
    #N = int(N)
    M = 512
    N = 512
    K = 224
    random.seed(30)
    target = 'llvm -mcpu=core-avx2'
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

    numpyBaseline(M,K,N)

    s = tvm.create_schedule(C.op)
    print("schedule base")
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    s = Gemm_v0(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    s = Gemm_v1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    s = Gemm_v2(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    s = Gemm_v3(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    s = Gemm_v4(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    s = Gemm_v5(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    s = Gemm_v6(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

 ##################################################################################################
    # Summary
    # -------
    # After applying the above simple optimizations with only 18 lines of code,
    # our generated code can achieve 60% of the `numpy` performance with MKL.
    # Note that the outputs on the web page reflect the running times on a non-exclusive
    # Docker container, thereby they are *unreliable*. It is highly encouraged to run the
    # tutorial by yourself to observe the performance gain acheived by TVM.
