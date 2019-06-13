import tensorflow as tf
import numpy as np
import tensorflow.layers as layers

inputs = np.ones([32, 100])
x = tf.placeholder(tf.float32, [None, 100])
print(x.shape.ndims)
y = layers.dense(x, 10)

print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
