import tensorflow as tf


a = tf.Variable([1, 2, 3])
b = tf.get_variable("bbb", shape=[2,3])
tf.constant([3])
c = tf.ones([3])
d = tf.random_uniform([3, 4])
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
