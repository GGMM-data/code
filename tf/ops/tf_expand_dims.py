import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.int32, [None, 10])
y1 = tf.expand_dims(x, 0)
y2 = tf.expand_dims(x, 1)
y3 = tf.expand_dims(x, 2)
y4 = tf.expand_dims(x, -1) # -1表示最后一维
# y5 = tf.expand_dims(x, 3) error

with tf.Session() as sess:
   inputs = np.random.rand(12, 10)
   r1, r2, r3, r4 = sess.run([y1, y2, y3, y4], feed_dict={x: inputs})
   print(r1.shape)
   print(r2.shape)
   print(r3.shape)
   print(r4.shape)
