import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.float32, shape=None)

with tf.Session() as sess:
    inputs = np.arange(100)
    for i in range(100):
        out = sess.run(x, feed_dict={x: inputs[i]})
        print(out)
