import tensorflow as tf
import numpy as np


def some_func(sess):


   x = tf.placeholder(tf.int32, [10])
   y = tf.constant([10, 3.2])

   inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
   for i in range(10):
      print(sess.run(tf.equal(x[i], 0), feed_dict={x: inputs}))
      if tf.equal(x[i], 0):
         print(True)
         y = tf.add(y, 1)
      else:
         print(False)
         y = tf.add(y, 10)

   result = tf.log(y)

   sess.run(result, feed_dict={x: inputs})


sess = tf.Session()
some_func(sess)

