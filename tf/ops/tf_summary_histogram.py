import tensorflow as tf
import numpy as np

shape = 2
x = tf.placeholder(tf.float32, shape=[shape])
x_sum = tf.summary.histogram("x", x)
summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter("tf_summary/histogram/")
# Setup a session and summary writer
with tf.Session() as sess:
    N = 3
    for step in range(N):
        print(step)
        # inputs = np.arange(shape) + 5*step
        inputs = np.ones(shape) + 5*step
        outputs, summ = sess.run([x, x_sum], feed_dict={x: inputs})
        print(outputs)
        print(type(summ))
        writer.add_summary(summ, global_step=step)

