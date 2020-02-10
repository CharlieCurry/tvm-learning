import logging
import numpy as np
import tvm
import random
import sys
import math
import timeit
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
    func = tvm.build(s, [A, B, C], target=target, name='mmult')
    assert func
    func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    print('time: %f' % evaluator(a, b, c).mean)
    #print(tvm.lower(s, [A, B, C], simple_mode=True))

def Gemm_v1(C):
    # Blocking
    # --------
    # A important trick to enhance the cache hit rate is blocking --- data chunk will be computed
    # block by block. The memory access inside the block is a small neighbourhood which is with high
    # memory locality. In this tutorial, I picked up 32 as the blocking factor. So the block will
    # fill 32 * 32 * sizeof(float) which is 4KB in the cache whose total size is 32KB (L1 data cache)
    s = tvm.create_schedule(C.op)
    bn = 32
    # Blocking by loop tiling
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    # Hoist reduction domain outside the blocking loop
    s[C].reorder(xo, yo, ko, ki, xi, yi)
    return s

def Gemm_v2(C):
    bn = 32
    s = tvm.create_schedule(C.op)
    ###################################################################################################
    # Vectorization
    # -------------
    # Another important trick is vectorization. When the memory access pattern is uniform,
    # the compiler can detect this pattern and pass the continuous memory to vector processor. In TVM,
    # we can use `vectorize` interface to hint the compiler this pattern, so that we can accelerate it vastly.
    #
    # In this tutorial, we chose to vectorize the inner loop row data since it is cache friendly.
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    s[C].reorder(xo, yo, ko, ki, xi, yi)
    # Vectorization
    s[C].vectorize(yi)
    return s

def Gemm_v3(C):
    s = tvm.create_schedule(C.op)
    ###################################################################################################
    # Loop Permutation
    # ----------------
    # If we look at the above IR, we can see the inner loop row data is vectorized and
    # B is transformed into PackedB. The traversal of PackedB is sequential now.
    # So we will look at the access pattern of A. In current schedule, A is accessed column by column
    # which is not cache friendly. If we change the nested loop order of ki and inner axes xi,
    # the access pattern for A matrix is more cache friendly.
    bn = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    # re-ordering
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    return s

def Gemm_v4(C):
    s = tvm.create_schedule(C.op)
    bn = 32
    ###################################################################################################
    # Array Packing
    # -------------
    # Another important trick is array packing. This trick is to reorder the storage dimension of the
    # array to convert the continuous access pattern on certain dimension to a sequential pattern after
    # flattening.
    #
    # .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/array-packing.png
    #      :align: center
    #      :scale: 100%
    #
    ###################################################################################################
    # Just as it is shown in the figure above, after blocking the computations, we can observe the array
    # access pattern of B (after flattening), which is regular but discontinuous. We expect that after
    # some transformation we can get continuous access pattern. We can reorder a [16][16] array to
    # a [16/4][16][4] array, so that the access pattern of B will be sequential when grabing
    # the corresponding value from the packed array.
    # We have to re-write the algorithm slightly.
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
    return s

def Gemm_v5(C):
    bn = 32
    s = tvm.create_schedule(C.op)
    k, = s[C].op.reduce_axis
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k),
                    name='C')
    s = tvm.create_schedule(C.op)
    ################################################################################################
    # Write cache for blocks
    # ----------------------
    # After blocking, the program will write result to C block by block, the access pattern
    # is not sequential. So we can use a sequential cache array to hold the block results and
    # write to C when all the block results are ready.
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
    return s

def Gemm_v6(C):
    bn = 32
    s = tvm.create_schedule(C.op)
    k, = s[C].op.reduce_axis
    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, tvm.indexmod(y, bn)], axis=k),
                    name='C')
    s = tvm.create_schedule(C.op)
    ##################################################################################################
    # Parallel
    # --------
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
    return s


if __name__ == '__main__':
    # M = sys.argv[1]
    # K = sys.argv[2]
    # N = sys.argv[3]
    # M = int(M)
    # K = int(K)
    # N = int(N)
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
