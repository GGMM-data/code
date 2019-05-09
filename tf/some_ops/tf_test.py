import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.int32, [10,])
y = tf.constant([10, 3.2])

for i in range(10):
   if x[i] != 0 :
      tf.add(y, 1)
   else:
      tf.sub(y, 1) 
   
add = tf.log(y)

with tf.Session() as sess:
   inputs = np.array([1, 2,3,4,5,6,7,8,9,0])
   print(sess.run(add, feed_dict={x: inputs}))
