
import tensorflow as tf

sess = tf.Session()
saver = tf.train.import_meta_graph('./checkpoint_dir/TSSModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))

##Model has been restored. Above statement will print the saved value
# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13,w2:17}

#接下来，访问你想要执行的op
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print(sess.run(op_to_restore,feed_dict))
