import logging
import numpy as np
import tvm
import random
import sys
import timeit
from tvm import relay
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
    print('time: %.10f' %result)
    #print(tvm.lower(s, [A, B, C], simple_mode=True))
    return result

def testwithNoneopt(str,ctx,method):
    # apply history best from log file
    autotvm.record.pick_best(str,str+".best")
    with autotvm.apply_history_best(str):
        with tvm.target.create("llvm"):
            s, arg_bufs = method(N, K, M, 'float32')
            func = tvm.build(s, arg_bufs)
    # check correctness
    a_np = np.random.uniform(size=(N, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, M)).astype(np.float32)
    c_np = a_np.dot(b_np)
    c_tvm = tvm.nd.empty(c_np.shape)
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-5)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    print(str)
    print('TVM: %f' % evaluator(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm).mean)


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
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    ko, ki = cfg["tile_k"].apply(s, C, k)
    '''
    >>> # use custom candidates
    >>> cfg.define_split('tile_x', x, policy='candidate', candidate=[[1, 4, 4], [4, 1, 4]])'''
    #>>> # use a filter that only accepts the split scheme whose inner most tile is less then 64
    # cfg.define_split('tile_x', x, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_y', y, policy='factors', filter=lambda x: x.size[-1] <= 64)
    # cfg.define_split('tile_k', k, policy='factors', filter=lambda x: x.size[-1] <= 64)

    # cfg.define_knob("tile_x", [1, 4, 8, 16, 32])
    # cfg.define_knob("tile_y", [1, 4, 8, 16, 32])
    # cfg.define_knob("tile_k", [1, 4, 8, 16, 32])
    # print(cfg['tile_x'])
    # xo, xi = s[C].split(x, cfg['tile_x'].val)
    # yo, yi = s[C].split(y, cfg['tile_y'].val)
    # ko, ki = s[C].split(k, cfg['tile_k'].val)
    s[C].reorder(xo,yo,ko,xi,ki,yi)
    s[C].vectorize(yi)
    s[C].parallel(xo)
    return s, [A, B, C]


if __name__ == '__main__':
    M = sys.argv[1]
    K = sys.argv[2]
    N = sys.argv[3]
    M = int(M)
    K = int(K)
    N = int(N)
    print(str(M)+"*"+str(K)+"*"+str(N))
    random.seed(30)
    target = 'llvm'
    dtype = 'float32'
    ctx = tvm.context(target, 0)

    a_np = np.random.rand(M,K).astype(dtype)
    b_np = np.random.rand(K,N).astype(dtype)
    c_np = np.zeros((M,N)).astype(dtype)
    c_np = a_np.dot(b_np)

    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(c_np, ctx)


    s,[A,B,C] = Gemm_tv2_reorder2_3_vec1_para1_config_define(N,K,M,dtype)
    print(tvm.lower(s, [A, B, C], simple_mode=True))