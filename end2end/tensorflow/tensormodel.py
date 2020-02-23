import tvm
from tvm import relay
import numpy as np
import os.path
import tensorflow as tf
import tvm.relay.testing.tf as tf_testing
from tvm.contrib.download import download_testdata
def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

def run_inference_on_image(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path,uid_lookup_path=label_path)

        # Print top 5 predictions from tensorflow.
        top_k = predictions.argsort()[-5:][::-1]
        print ("===== TENSORFLOW RESULTS =======")
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))



#repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'

target = 'llvm'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

img_path = 'elephant-299.jpg'
model_path = 'classify_image_graph_def-with_shapes.pb'
map_proto_path = 'imagenet_2012_challenge_label_map_proto.pbtxt'
label_path = 'imagenet_synset_to_human_label_map.txt'

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')

######################################################################
# Decode image
# ------------
# .. note::
#
#   tensorflow frontend import doesn't support preprocessing ops like JpegDecode.
#   JpegDecode is bypassed (just return source node).
#   Hence we supply decoded frame to TVM instead.
#

from PIL import Image
image = Image.open(img_path).resize((299, 299))

x = np.array(image)

######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}

mod, params = relay.frontend.from_tensorflow(graph_def,layout=layout,shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")
######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,target=target,target_host=target_host,params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.

from tvm.contrib import graph_runtime
dtype = 'uint8'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('DecodeJpeg/contents', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), 'float32'))

######################################################################
# Process the output
# ------------------
# Process the model output to human readable text for InceptionV1.
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path,uid_lookup_path=label_path)

# Print top 5 predictions from TVM output.
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print('%s (score = %.5f)' % (human_string, score))

######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow
run_inference_on_image(img_path)

