import tensorflow as tf
import time
import numpy as np


x_p = tf.placeholder(tf.float32, shape=None, name="x")
# y = tf.Variable(x_p)
y = x_p
s_y = tf.summary.scalar("y", y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./tf_summary/")

    flag = True
    count = 1
    x = np.arange(4000)
    while True:
            begin = time.time()
            _y, _sy = sess.run([y, s_y], feed_dict={x_p: x[count]})
            writer.add_summary(_sy, global_step=count)
            count += 1
            if count >= 4000:
                break
            print(count, time.time() - begin)
    
