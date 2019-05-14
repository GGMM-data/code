import tensorflow as tf

a = tf.Variable([1, 2, 3])
b = tf.get_variable("bbb", shape=[2,3])
tf.constant([3])
c = tf.ones([3])
d = tf.random_uniform([3, 4])
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
# [<tf.Variable 'Variable:0' shape=(3,) dtype=int32_ref>, <tf.Variable 'bbb:0' shape=(2, 3) dtype=float32_ref>]
# 可以看出来，只有tf.Variable()和tf.get_variable()产生的变量会加入到这个图中
