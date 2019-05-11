import tensorflow as tf
import numpy as np


# Example 1
x = tf.placeholder(tf.int32, [10])
y = tf.constant([10, 3.2])


for i in range(10):
    y = tf.cond(tf.equal(x[i], 0), lambda: tf.add(y, 1), lambda: tf.add(y, 10))

# for i in range(10):
#     if x[i] == 0:
#         y = tf.add(y, 1)
#     else:
#         y = tf.add(y, 10)

result = tf.log(y)

with tf.Session() as sess:
   inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
   print(sess.run(result, feed_dict={x: inputs}))


# Example 2
def myfunc(x):
   if (x > 0):
      return 1
   return 0


with tf.Session() as sess:
    x = tf.constant(4)
    # print(myfunc(x))
    # raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
    # TypeError: Using a `tf.Tensor` as a Python `bool` is not allowed. Use `if t is not None:` instead of `if t:` to test if a tensor is defined, and use TensorFlow ops such as tf.cond to execute subgraphs conditioned on the value of a tensor.
    result = tf.cond(tf.greater(x, 0), lambda: 1, lambda: 0)
    print(type(result))
    print(result.eval())

# Example 3
x = tf.constant(4)
y = tf.constant(4)

with tf.Session() as sess:
    print(x) 
    print(y) 
    if x == y:
      print(True)
    else:
      print(False)
    result = tf.equal(x, y)
    print(result.eval())
    def f1(): 
      print("f1 declare")
      return [1, 1]
    def f2():
      print("f2 declare")
      return [0, 0]
    res = tf.cond(tf.equal(x, y), f1, f2)
    print(res)
