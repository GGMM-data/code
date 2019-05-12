import numpy as np
import tensorflow as tf


X = tf.placeholder(tf.int32, [None, 7])

zeros = tf.zeros_like(X)
index = tf.not_equal(X, zeros)
loc = tf.where(index)

with tf.Session() as sess:
    inputs = np.array([[1, 0, 3, 5, 0, 8, 6], [2, 3, 4, 5, 6, 7, 8]])
    out = sess.run(loc, feed_dict={X: inputs})
    print(np.array(out))
    # 输出12个坐标，表示这个数组中不为0元素的索引。



inputs = np.array([[1, 0, 3, 5, 0, 8, 6], [2, 3, 4, 5, 6, 7, 8]])
zeros = tf.zeros_like(X)
ones = tf.ones_like(X)
# index = tf.not_equal(X, zeros)
loc = tf.where(inputs, x=ones, y=zeros)

with tf.Session() as sess:
    out = sess.run(loc, feed_dict={X: inputs})
    print(np.array(out))
 
