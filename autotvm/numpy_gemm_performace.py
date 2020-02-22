import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

import numpy as np
import timeit

np.show_config()

repeat_numer = 2
def np_matmul_timer(N):
    np_timer = timeit.timeit(setup='import numpy\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "float64"\n'
                                     'a = numpy.random.rand(N, N).astype(dtype)\n'
                                     'b = numpy.random.rand(N, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=repeat_numer)
    return np_timer


# sizes = 2**np.arange(8, 18, 2)
sizes = [32,128,1024,2048,4096,6144]
exe_times = [np_matmul_timer(n) for n in sizes]

#注意这个gflops的计算方式
np_gflops = 2 * np.array(sizes) ** 3 / 1e9 / np.array(exe_times) * repeat_numer
print("size:",sizes)
print("exe_times:",np.array(exe_times)/repeat_numer)
print("np_gflops:",np_gflops)