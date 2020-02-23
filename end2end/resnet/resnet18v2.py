import os
import tvm
import onnx
from tvm import relay
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import numpy as np
from PIL import Image
import matplotlib as plt
import mxnet as mx
from mxnet.gluon.data.vision import transforms


def load_onnx_model(model_filename):
    onnx_model = onnx.load(model_filename)
    input_name = '1'  # change '1' to '0'
    shape_dict = {input_name:(1, 3, 224, 224)}
    sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return sym, params

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def export_tvm_model(modelname, lib, graph, params):
    export_path = os.path.join(".")

    lib_path = os.path.join(export_path, "%s.so" % modelname)
    lib.export_library(lib_path)

    with open(os.path.join(export_path, "%s.json" % modelname), "w") as fo:
        fo.write(graph.json())
    with open(os.path.join(export_path, "%s.params" % modelname), "wb") as fo:
        fo.write(tvm.nnvm.compiler.save_param_dict(params))

def tune_and_evaluate(net, params, input_shape, output_shape, tuning_opt):
    input_name = net.list_input_names()[0]
    print("Extracting tasks from graph.  Input is %s with shape %s" % (input_name, str(input_shape)))
    tasks = autotvm.task.extract_from_graph(net, target=target,
                                                 shape={input_name: input_shape}, dtype="float32",
                                                 symbols=tuple(tuneable))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.compiler.build_config(opt_level=3):
            graph, lib, params = tvm.nnvm.compiler.build(
                net, target=target, shape={'data': input_shape}, params=params, dtype={"reshape_attr_tensor164": "int64"})

        # export library
        tmp = tempdir()
        #filename = "net.tar"
        #lib.export_library(tmp.relpath(filename))
        export_tvm_model("%s-optimized" % model_name, lib, graph, params)
        print("exported optimized model")


        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


img_path = "cat.png"
image = Image.open(img_path)
image.show()
image = np.array(image)
print(image.shape)
batch_size = 1
model_name = 'resnet18v2'
log_file = "%s-batchsize%d-optimization.log" % (model_name, batch_size)

input_filename = "kitten.jpg"


tuning_options = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4),
    ),
}

tuneable = tvm.autotvm.task.get_config()

#### DEVICE CONFIG ####
target = tvm.target.cuda()
input_shape = (batch_size,3,224,224)
output_shape = (batch_size,1000)
sym, params = load_onnx_model(model_name + ".onnx")

tune_and_evaluate(sym, params, input_shape, output_shape, tuning_options)