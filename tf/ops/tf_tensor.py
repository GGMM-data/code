import tensorflow as tf


sess = tf.Session()

x = tf.constant([[3, 4], [5, 8]])
print(sess.run(tf.constant([3,4])))
print(sess.run(tf.ones_like(x)))
print(sess.run(tf.zeros_like(x)))
# https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/random_normal
print(sess.run(tf.random_normal([2,2])))
# tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None) https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/random_uniform
print(sess.run(tf.random_uniform([2,2])))
print(sess.run(tf.random_uniform([2,2], dtype=tf.int32, maxval=4)))
print(sess.run(tf.ones([3, 4])))
print(sess.run(tf.zeros([2,2])))

a = tf.Variable(tf.zeros([4]), trainable=False)
b = tf.Variable(tf.zeros([2, 2]))
print(sess.run(tf.global_variables_initializer()))
# print(sess.run(a))
print(sess.run(b))
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
print(sess.run(a))
op = a[0].assign(a[0]+1)
print(sess.run(op))

op_list = []
for i in range(a.shape[0]):
    op_list.append(a[i].assign(a[i]+1))

for i in range(a.shape[0]):
    print(sess.run(op_list[i]))

# [3 4]
# [[1 1]
#  [1 1]]
# [[0 0]
#  [0 0]]
# [[-0.5188188   0.77538687]
#  [ 1.2343276  -0.58534193]]
# [[0.8851745  0.12824357]
#  [0.28489232 0.76961493]]
# [[0 2]
#  [2 1]]
# [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]]
# [[0. 0.]
# 
