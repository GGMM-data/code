import tensorflow as tf
import numpy as np

number = 3
x_ph_list = []
for i in range(number):
    x_ph_list.append(tf.placeholder(tf.float32, shape=None))

x_summary_list = []
for i in range(number):
    x_summary_list.append(tf.summary.scalar("x%s" % i, x_ph_list[i]))

writer = tf.summary.FileWriter("./tf_summary/scalar_list_summary/sep")
with tf.Session() as sess:
    scope = 10
    inputs = np.arange(scope*number)
    inputs = inputs.reshape(scope, number)
    # inputs = np.random.randn(scope, number)
    for i in range(scope):
        for j in range(number):
            out, xj_s = sess.run([x_ph_list[j], x_summary_list[j]], feed_dict={x_ph_list[j]: inputs[i][j]})
            writer.add_summary(xj_s, global_step=i)
