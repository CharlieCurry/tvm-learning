# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-tuning a convolutional network for x86 CPU
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Eddie Yan <https://github.com/eqy>`_

This is a tutorial about how to tune convolution neural network
for x86 CPU.
"""
import os
import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime

import tensorflow as tf
#import tvm.relay.testing.tf as tf_testing
from PIL import Image
img_path='test.jpg'

import logging
# logging.getLogger('autotvm').setLevel(logging.DEBUG)

# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
target = "llvm"
batch_size = 1
dtype = "float32"
model_name = "eff"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name

# Set the input name of the graph
# For ONNX models, it is typically "0".
input_name = "input_x"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 48
os.environ["TVM_NUM_THREADS"] = str(num_threads)

#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can either load some pre-defined network from :code:`relay.testing`
# or building :any:`relay.testing.resnet` with relay.
# We can also load models from MXNet, ONNX and TensorFlow.
#
# In this tutorial, we choose resnet-18 as tuning example.


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    layout = 'NCHW'# 'NCHW'
    input_shape = (batch_size, 380, 380, 1) # layout = 'NCHW' but input is NHWC, is right, if input is not NHWC it will be error
    output_shape = (batch_size, 2)

    ctx = tvm.cpu(0)
    model_path = 'tf_test_tvm.pb'
    tf.reset_default_graph()
    with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        # Add shapes to the graph.
        with tf.compat.v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, 'logits')

    shape_dict = {'logits': (1,2)}
    # dtype_dict = {'DecodePng': 'uint8'}
    outputs = ['logits']
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 layout=layout,
    #                                              shape=shape_dict,
                                                 outputs=outputs
                                                 )

    return mod, params, input_shape, output_shape

#################################################################
# Configure tensor tuning settings and create tasks
# -------------------------------------------------
# To get better kernel execution performance on x86 CPU,
# we need to change data layout of convolution kernel from
# "NCHW" to "NCHWc". To deal with this situation, we define
# conv2d_NCHWc operator in topi. We will tune this operator
# instead of plain conv2d.
#
# We will use local mode for tuning configuration. RPC tracker
# mode can be setup similarly to the approach in
# :ref:`tune_relay_arm` tutorial.

tuning_option = {
    'log_filename': log_file,
    'tuner': 'xgb',
    'early_stopping': 100,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1,
                                   min_repeat_ms=1000),
    ),
}


# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):
    print('tasks: {}'.format(tasks))
    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        op_name = tsk.workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key='direct')
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        n_trial = 1 # set small num to quick test
        print('n trial {}'.format(n_trial))
#         print(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=False):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=10)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    # why "main"? 
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params, ops=(relay.op.nn.conv2d,))

#     # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)
    # compile kernels with graph-level best records
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)
#             graph, lib, params = relay.build(mod,
#                                      target=target,
#                                      target_host=target,
#                                      params=params)
            
        
        base_path = './lib'
        path_lib = os.path.join(base_path, "deploy_lib.so")
        lib.export_library(path_lib)
        with open(os.path.join(base_path, "deploy_graph.json"), "w") as fo:
            fo.write(graph)
        with open(os.path.join(base_path, "deploy_param.params"), "wb") as fo:
            fo.write(relay.save_param_dict(params))

        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=(1, 380, 380, 1))).astype(dtype))
#         data_tvm = preprocess(img_path)
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
        


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

def preprocess(img_path):
    image = Image.open(img_path)
    image.crop
    image = image.resize((380, 380), Image.BILINEAR)
    x = np.array(image)
    x = x / 255
    x = x - 0.5
    x = x * 2
    xx = np.expand_dims(x, 0)
    xx = np.expand_dims(xx, 0)
    print('input shape {}'.format(xx.shape))
    return xx

tune_and_evaluate(tuning_option)

