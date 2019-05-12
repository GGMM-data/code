import tensorflow as tf


a = tf.constant([1, 1])
b = tf.constant([2, 2])
c = tf.add(a, b)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(a))
    print(sess.run(b))
