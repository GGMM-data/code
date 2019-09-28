import tensorflow as tf


a = tf.Variable([1, 2, 3])
b = tf.get_variable("bbb", shape=[2,3])
tf.constant([3])
c = tf.ones([3])
d = tf.random_uniform([3, 4])
e = tf.log(c)

# 查看GLOBAL_VARIABLES collection中的变量
global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
for var in global_variables:
   print(var)

# 查看TRAINABLE_VARIABLES collection中的变量
trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for var in global_variables:
   print(var)
