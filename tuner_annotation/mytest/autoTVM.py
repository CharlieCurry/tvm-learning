import tvm
import numpy
import timeit
from tvm import autotvm
import logging
import sys

# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
#指定矩阵的大小(M, K) x (K, N)
import sys
# M = sys.argv[1]
# K = sys.argv[2]
# N = sys.argv[3]
# M = int(M)
# K = int(K)
# N = int(N)
M = 128
N = 128
K = 128
src = str(M)+"*"+str(K)+"*"+str(N)
print(src)
# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = 'llvm -mcpu=core-avx2'

#Construct a TVM context with given device type and id.
ctx = tvm.context(target, 0)#== tvm.cpu(0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)

#多次计算求平均
np_repeat = 100
#构建numpy的计算
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
    """Create a new IterVar for reduction.
     Parameters
     ----------
     dom : Range
         The domain of iteration.
     name : str
         The name of the variable.
     Returns
     -------
     axis : IterVar
         An iteration variable representing the value.
     """
    A = tvm.placeholder((M, K), name='A')
    B = tvm.placeholder((K, N), name='B')

    ##### define space begin #####
    cfg = autotvm.get_config()
    """Get current config object
      Returns
      -------
      cfg: ConfigSpace or ConfigEntity
          The current config
      """
    cfg.define_split("tile_x", M, num_outputs=2)
    cfg.define_split("tile_y", N, num_outputs=2)
    cfg.define_split("tile_k", K, num_outputs=2)
    """Define a new tunable knob which splits an axis into a list of axes
    Parameters
    ----------
    name: str
        name to index the entity of this space
    axis: tvm.schedule.IterVar
        axis to split
    policy: str
        name of policy.
        If is 'factors', the tuner will try all divisible factors.
        If is 'power2', the tuner will try power-of-two factors less or equal to the length.
        If is 'verbose', the tuner will try all candidates in above two policies.
        If is 'candidate', try given candidates.
    kwargs: dict
        extra arguments for policy
        max_factor: int
            the maximum split factor.
        filter: function(int) -> bool
            see examples below for how to use filter.
        num_outputs: int
            the total number of axis after split.
        no_tail: bool
            should we only include divisible numbers as split factors.
        candidate: list
            (policy=candidate) manual candidate list.

    Examples
    --------
    >>> # use custom candidates
    >>> cfg.define_split('tile_x', x, policy='candidate', candidate=[[1, 4, 4], [4, 1, 4]])

    >>> # use a filter that only accepts the split scheme whose inner most tile is less then 4
    >>> cfg.define_split('tile_y', y, policy='factors', filter=lambda x: x.size[-1] <= 4)
    """

    ##### define space end #####

    # We have to re-write the algorithm slightly.
    #print("cfg[tile_y]",cfg["tile_y"])#打印tile_y的候选空间,如[-1,128]
    xn = cfg["tile_x"].size[-1]
    bn = cfg["tile_y"].size[-1]#只打印列表里的最后一个，如上面的128
    kn = cfg["tile_k"].size[-1]
    #print("xn:",xn,"bn:",bn,"kn:",kn)


    packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    """Construct a new tensor by computing over the shape domain.
        The compute rule is result[axis] = fcompute(axis)
        Parameters
        ----------
        shape: Tuple of Expr
            The shape of the tensor
        fcompute: lambda function of indices-> value
            Specifies the input source expression
        name: str, optional
            The name hint of the tensor
        tag: str, optional
            Additional tag information about the compute.
        attrs: dict, optional
            The additional auxiliary attributes about the compute.
        Returns
        -------
        tensor: Tensor
            The created tensor
        """
    #" // " 表示整数除法,返回不大于结果的一个最大的整数
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y // bn, k, y % bn], axis=k),
                    name = 'C')

    s = tvm.create_schedule(C.op)
    """Create a schedule for list of ops
      Parameters
      ----------
      ops : list of Operations
          The source expression.
      Returns
      -------
      sch : schedule.Schedule
          The created schedule.
      """
    x, y = s[C].op.axis
    k, = s[C].op.reduce_axis
    #print("x:", (x))#x: iter_var(x, range(min=0, ext=1024))

    # schedule according to config
    # Allocate write cache
    CC = s.cache_write(C, 'global')
    '''
    在存储到tensor之前，创建原始tensor的缓存写入。这会使张量体发生变异。
在传入张量之前，将创建一个新的缓存阶段。此函数可用于支持数据布局转换。
如果在张量的数据平行轴上存在分裂/融合/重新排序在调用缓存写入之前。中间缓存存储
布局中的数据作为离开轴的迭代顺序。数据将转换回原始张量中的原始布局。用户可以进一步调用
compute_inline以内联原始布局并保持存储在转换后的布局中的数据。
 Parameters
        ----------
        tensor : Tensor, list or tuple
            The tensors to be feed to. All the tensors must be produced by one computeOp
        scope : str
            The scope of cached
        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
    '''
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    s[C].reorder(xo, yo, xi, yi)

    # Write cache is computed at yo
    s[CC].compute_at(s[C], yo)
    """Attach the stage at parent's scope
           Parameters
           ----------
           parent : Stage
               The parent stage
           scope : IterVar
               The loop scope t be attached to.
           """
    # New inner axes
    xc, yc = s[CC].op.axis
    k, = s[CC].op.reduce_axis

    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    """Unroll the iteration.
            Parameters
            ----------
            var : IterVar
                The iteration to be unrolled.
            """

    s[CC].vectorize(yc)
    """Vectorize the iteration.
           Parameters
           ----------
           var : IterVar
               The iteration to be vectorize
           """
    # parallel
    s[C].parallel(xo)
    """Parallelize the iteration.
          Parameters
          ----------
          var : IterVar
              The iteration to be parallelized.
          """
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    return s, [A, B, C]

task = autotvm.task.create(matmul, args=[], target=target)
measure_option = autotvm.measure_option(
    builder='local',
#Run generated code on local devices.
    runner=autotvm.LocalRunner(number=5))
"""auto.measure_option:
   Set options for measure. To measure a config, we will build it and run it.
   So we have to set options for these two steps.
   They have their own options on timeout, parallel, etc.
   Parameters
   ----------
   builder: Builder
       Specify how to build programs
   runner: Runner
       Specify how to run programs
   Examples
   --------
   # example setting for using local devices
   >>> measure_option = autotvm.measure_option(
   >>>     builder=autotvm.LocalBuilder(),      # use all local cpu cores for compilation
   >>>     runner=autotvm.LocalRunner(          # measure them sequentially
   >>>         number=10,
   >>>         timeout=5)
   >>> )

   # example setting for using remote devices
   >>> measure_option = autotvm.measure_option(
   >>>    builder=autotvm.LocalBuilder(),  # use all local cpu cores for compilation
   >>>    runner=autotvm.RPCRunner(
   >>>        'rasp3b', 'locahost', 9190, # device key, host and port of the rpc tracker
   >>>        number=4,
   >>>        timeout=4) # timeout of a run on the device. RPC request waiting time is excluded.
   >>>)

   Note
   ----
   To make measurement results accurate, you should pick the correct value for the argument
   `number` and `repeat` in Runner(). Some devices need a certain minimum running time to
   "warm up," such as GPUs that need time to reach a performance power state.
   Using `min_repeat_ms` can dynamically adjusts `number`, so it is recommended.
   The typical value for NVIDIA GPU is 150 ms.
   """

"""autotvm.LocalRunner:
    Run generated code on local devices.
    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float, optional
        The cool down interval between two measurements.
    check_correctness: bool, optional
        Whether check correctness after measurement. This will use llvm cpu target to
        call your template and get the reference output.
        This can work for TOPI templates, but may not work for your custom template.
    Note
    ----
    This is a "fake" local mode. We start a silent rpc tracker and rpc server
    for the user. In this way we reuse timeout/isolation mechanism in RPC infrastructure.
    """
# begin tuning, log records to file `matmul.log`
tuner = autotvm.tuner.XGBTuner(task)
# tuner有众多的参数
n_trial = 200
early_stopping = 200
tuner.tune(n_trial=n_trial,
           early_stopping=early_stopping,
           measure_option=measure_option,
           callbacks=[autotvm.callback.progress_bar(n_trial),
                      autotvm.callback.log_to_file('matmul'+src+'.log.tmp')])


# pick best records to a cache file
autotvm.record.pick_best("matmul"+src+".log.tmp", 'matmul'+src+'.log')
"""
    Pick best entries from a file and store it to another file.
    This distill the useful log entries from a large log file.
    If out_file already exists, the best entries from both
    in_file and out_file will be saved.

    Parameters
    ----------
    in_file: str
        The filename of input
    out_file: str or file
        The filename of output
    """
with autotvm.apply_history_best('matmul'+src+'.log'):
    with tvm.target.create("llvm -mcpu=core-avx2"):
        s, arg_buf = matmul()
        func = tvm.build(s, arg_buf)
# func = tvm.build(s, arg_buf, target=target, name='mmult')
assert func

func(a, b, c)
#测试两者的数据正确性
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
""" Version of np.testing.assert_allclose with `atol` and `rtol` fields set
  in reasonable defaults.
  Arguments `actual` and `desired` are not interchangable, since the function
  compares the `abs(actual-desired)` with `atol+rtol*abs(desired)`.  Since we
  often allow `desired` to be close to zero, we generally want non-zero `atol`.
  """
#print(func.get_source("asm"))

evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
print('TVM:',evaluator(a, b, c).mean)