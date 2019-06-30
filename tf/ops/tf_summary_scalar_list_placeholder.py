import tensorflow as tf
import numpy as np


number = 3
x_summary = []
x = tf.placeholder(tf.float32, shape=[number])
for i in range(number):
    x_summary.append(tf.summary.scalar("x%s" % i, x[i]))

merged_summary = tf.summary.merge(x_summary, "x_summary")
writer = tf.summary.FileWriter("./tf_summary/scalar_list_summary")
with tf.Session() as sess:
    scope = 10
    inputs = np.arange(scope*number)
    inputs = inputs.reshape(scope, number)
    # inputs = np.random.randn(scope, number)
    for i in range(scope):
        out, x_s = sess.run([x, merged_summary], feed_dict={x: inputs[i]})
        writer.add_summary(x_s, global_step=i)
