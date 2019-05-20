import tensorflow as tf


v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

b = tf.Variable([3, 4.0])
c = tf.Variable([3, 4.0])

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# 初始化op
init_op = tf.global_variables_initializer()

# 创建saver
saver1 = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run([inc_v1, dec_v2])
    save_path1 = saver1.save(sess, "saver/variables/all_variables.ckpt")
    print("All variable save in path: %s" % save_path1)

