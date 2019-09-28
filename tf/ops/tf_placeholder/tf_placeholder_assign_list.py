import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[3])

with tf.Session() as sess:
    inputs = [1, 2, 3]
    outputs = sess.run(x, feed_dict={x: inputs})
    print(outputs)
