import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

v1 = tf.Variable([1, 2], name="v1")
v2 = tf.Variable(44.3, name="v2")

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
checkpoint_path = './checkpoint/saver2.ckpt'

sess.run(init_op)
print(sess.run(v1))
print(sess.run(v2))
save_path = saver.save(sess, checkpoint_path)
print("Model saved in files: %s" % save_path)

