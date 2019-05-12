import tensorflow as tf
import numpy as np

sess = tf.Session()

# Example 1
data = np.array([[0, 1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10, 11],
          [12, 13, 14, 15, 16, 17],
          [18, 19, 20, 21, 22, 23],
          [24, 25, 26, 27, 28, 29]])
data = np.reshape(np.arange(30), [5, 6])
x = tf.constant(data)
# Collecting elements from a tensor of rank 2
print(sess.run(x))
result = tf.gather_nd(x, [1, 2])
print(sess.run(result))
result = tf.gather_nd(x, [[1, 2], [2,3]])
print(sess.run(result))
# Collecting rows from a tensor of rank 2
result = tf.gather_nd(x, [[1],[2]])
print(sess.run(result))

# Example 2
data = np.array([[[0, 1],
          [2, 3],
          [4, 5]],
         [[6, 7],
          [8, 9],
          [10,11]]])
data = np.reshape(np.arange(12), [2, 3, 2])
x = tf.constant(data)
print(sess.run(x))
# Collecting elements from a tensor of rank 3
print(sess.run(x))
result = tf.gather_nd(x, [[0, 0, 0], [1, 2, 1]])
# Collecting batched rows from a tensor of rank 3
print(sess.run(result))
result = tf.gather_nd(x, [[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
print(sess.run(result))
result = tf.gather_nd(x, [[0, 0], [0, 1], [1, 0], [1, 1]])
print(sess.run(result))


# Example 3
