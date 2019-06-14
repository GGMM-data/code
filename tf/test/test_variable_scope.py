import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 10, 4], name="xx")
y = []
with tf.variable_scope("mxxhcm", reuse=False):
    for i in range(4):
        if i == 0:
            out = tf.layers.dense(x[:, :, i], 3, reuse=False)
            #out = tf.layers.dense(x[:, :, i], 3, name="first", reuse=False)
        else:
            out = tf.layers.dense(x[:, :, i], 3, reuse=True)
            #out = tf.layers.dense(x[:, :, i], 3, name="first", reuse=True)
        y.append(out)
init_op = tf.global_variables_initializer()
inputs = np.random.rand(32, 10, 4)

for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(var)
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(4):
        outputs = sess.run(y, feed_dict={xx: inputs})
        print(len(outputs))

    
