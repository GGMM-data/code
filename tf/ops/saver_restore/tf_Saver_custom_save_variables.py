import tensorflow as tf


vv3 = tf.get_variable("v3", shape=[5], initializer=tf.zeros_initializer)
add_v3 = vv3.assign(vv3+3)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver({"v3": vv3})

with tf.Session() as sess:
    sess.run([add_v3])
    save_path = saver.save(sess, "saver/variables/custom_variables.ckpt")
    print("Custom variable save in path: %s" % save_path)
