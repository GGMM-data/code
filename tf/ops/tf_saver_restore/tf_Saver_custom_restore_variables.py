import tensorflow as tf


vv4 = tf.get_variable("v3", shape=[5])
saver = tf.train.Saver({"v3": vv4})
with tf.Session() as sess:
    saver.restore(sess, "saver/variables/custom_variables.ckpt")
    print(sess.run(vv4))
