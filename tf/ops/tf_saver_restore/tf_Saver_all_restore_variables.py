import tensorflow as tf

x1 = tf.get_variable("v1", shape=[3])
x2 = tf.get_variable("v2", shape=[5])
m = tf.get_variable("Variable", shape=[2])
n = tf.get_variable("Variable_1", shape=[2])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "saver/variables/all_variables.ckpt")
    print("Model restored")
    print(sess.run(x1))
    print(sess.run(x2))
    print(sess.run(m))
    print(sess.run(n))


