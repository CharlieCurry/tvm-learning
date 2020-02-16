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
    print('time: %f' %result)
    #print(tvm.lower(s, [A, B, C], simple_mode=True))
    return result

def Gemm_tv0(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    print("tv0:对最外两层进行tile")
    return s

def Gemm_tv1(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    # #tile接口只接受二维平铺
    # xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对最内轴进行划分
    #推荐使用
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=2,nparts =None)
    print("tv1:对最内层进行tile")
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    return s

def Gemm_tv2(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    print("tv2:对每层都进行tile")
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    return s

def Gemm_tv2_reoderv0(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    print("tv2_reoderv0:将o放置到i前，并保持内外相应的轴顺序不变")
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #将*o的调整到*i的前面是reorder的自然做法(实际上这几个顺序可以进行排列：o是不需要换的？，i是有必要换的！其实空间应该就为3！)
    s[C].reorder(xo, yo, ko, xi,yi,ki)
    return s

def Gemm_tv2_reoderv1_1(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, ko, yo, xi,yi,ki)
    return s

def Gemm_tv2_reoderv1_2(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(ko, yo, xo, xi,yi,ki)
    return s

def Gemm_tv2_reoderv1_3(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(yo, ko, xo, xi,yi,ki)
    return s

def Gemm_tv2_reoderv2_1(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, yo, ko, ki,yi,xi)
    return s

def Gemm_tv2_reoderv2_2(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, yo, ko, ki,xi,yi)
    return s

def Gemm_tv2_reoderv2_3(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, yo, ko, xi,ki,yi)
    return s

def Gemm_tv2_reoderv2_4(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, yo, ko, xi,yi,ki)
    return s

def Gemm_tv2_reoderv2_5(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, yo, ko, yi,xi,ki)
    return s

def Gemm_tv2_reoderv2_6(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, yo, ko, yi,ki,xi)
    return s

def Gemm_tv2_reorder2_3_vec1(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
    #tile接口只接受二维平铺
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    #通过这种方式对第三维的K进行划分
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=32,nparts =None)
    #注意这里的bn和factor都对应的是axis的inner的取值范围
    #这里我们试试o顺序变了有啥变化没
    s[C].reorder(xo, yo, ko, xi,ki,yi)
    #实验表明（无论从可行性和性能上都）应该针对最内层
    s[C].vectorize(yi)
    #s[C].parallel(xo)
    return s

def Gemm_tv2_reorder2_3_vec1_para1(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
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
    return s

def Gemm_tv2_reorder2_3_vec1_para1_unrollv1(C):
    # unroll
    # ki(best)
    # time: 0.000791
    # yi
    # time: 0.006034
    # xi
    # time: 0.001161
    # xo
    # time: 0.001919
    # ko(better)
    # time: 0.001041
    # yo
    #???
    s = tvm.create_schedule(C.op)
    bn = 32
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
    return s

def Gemm_tv2_reorder2_3_vec1_para1_unrollv2(C):
    # Tiling
    s = tvm.create_schedule(C.op)
    bn = 32
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
    s[C].unroll(xo)
    return s

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

def find_plan(s, A, B, C, a, b, c, ctx, c_np):
    print("numpy baseline")
    numpyBaseline(M, K, N)

    print("schedule baseline")
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("三种分块方式：")
    s = Gemm_tv0(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("在分块基础上加上reorder")
    s = Gemm_tv2_reoderv0(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("不同的outer:")
    s = Gemm_tv2_reoderv1_1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2_reoderv1_2(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2_reoderv1_3(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("不同的inner:")
    s = Gemm_tv2_reoderv2_1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2_reoderv2_2(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2_reoderv2_3(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2_reoderv2_4(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2_reoderv2_5(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2_reoderv2_6(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("vectorize:")
    s = Gemm_tv2_reorder2_3_vec1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("parallel")
    s = Gemm_tv2_reorder2_3_vec1_para1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

    print("unroll")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv2(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("packing")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("write cache")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1_writecachev1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)

def find_bestSchedule(s, A, B, C, a, b, c, ctx, c_np):
    print("################################################################")
    print("1.tiling")
    s = Gemm_tv2(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("2.reordering")
    s = Gemm_tv2_reoderv2_3(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("3.vectorize:")
    s = Gemm_tv2_reorder2_3_vec1(C)
    buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("4.parallel")
    s = Gemm_tv2_reorder2_3_vec1_para1(C)
    result4 = buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("5.unroll")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1(C)
    result5 = buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("6.packing")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1(C)
    result6 = buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("7.write cache")
    s = Gemm_tv2_reorder2_3_vec1_para1_unrollv1_packv1_writecachev1(C)
    result7 = buildandevaluation(s, A, B, C, a, b, c, ctx, c_np)
    print("################################################################")
    print("gemm可选方案：4，5，6，7")
    mintime = min(result4, result5, min(result6, result7))
    print("mintime=", mintime)
    print("speedup=", numpyBaseline(M, K, N) / mintime)
    print("################################################################")

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
    print(k)
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

    #find_plan(s, A, B, C, a, b, c, ctx, c_np)
    find_bestSchedule(s, A, B, C, a, b, c, ctx, c_np)
    

