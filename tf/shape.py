import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=None)
print(x.shape)
z = x 


with tf.Session() as sess:
    inputs = [np.random.rand()] 
    # print(inputs.shape)
    print(sess.run(z, feed_dict={x: inputs}))
