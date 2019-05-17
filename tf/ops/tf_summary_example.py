import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.float32, [5])
#sum_x = tf.summary.scalar('test x', x)
sum_x = tf.summary.histogram('test x', x)

with tf.Session() as sess:
    steps = 1
    writer = tf.summary.FileWriter("./summary/histogram", sess.graph)
    #inputs = np.array([[1, 1, 1, 1, 1], [2, 3, 4, 3, 2]]) 
    inputs = np.array([1, 1, 1, 1, 1]) 
    for i in range(steps, 2):
       inputs = inputs + i
       results, = sess.run([sum_x], feed_dict={x: inputs})
       print(results)
       writer.add_summary(results, i)

    writer.close()
   
