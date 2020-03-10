import tensorflow as tf

w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1= tf.Variable(2.0,name="bias")

w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name="op_to_restore")
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


saver.save(sess, './checkpoint_dir/TSSModel')