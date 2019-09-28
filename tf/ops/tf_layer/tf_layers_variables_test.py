import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

inputs = np.ones([32, 100])
layers.fully_connected(inputs, 10)

print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
