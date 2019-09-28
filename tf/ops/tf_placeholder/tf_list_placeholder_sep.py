import tensorflow as tf
import numpy as np

number = 3
x_ph_list = [] 
for i in range(number):
    x_ph_list.append(tf.placeholder(tf.float32, shape=None, name="x"+str(i)))

with tf.Session() as sess:
    inputs = np.arange(number)
    for i in range(number):
        out = sess.run(x_ph_list[i], feed_dict={x_ph_list[i]: inputs[i]})
        print(out)
