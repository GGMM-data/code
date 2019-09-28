import tensorflow as tf
import numpy as np


x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.log(x)
loss = tf.reduce_sum(y)

with tf.Session() as sess :
    # inputs = tf.constant([1.0, 2.0])
    inputs = np.array([[1.0, 2.0], [3, 4]])
    l = sess.run(loss, feed_dict={x: inputs})
    print(l)
