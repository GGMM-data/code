import tensorflow as tf

# 你所定义的变量一定要在 checkpoint 中存在；但不是所有在checkpoint中的变量，你都要重新定义。

config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

v1 = tf.Variable([33, 432], name="v1")
v2 = tf.Variable(2.0, name = "v2")

init_op = tf.global_variables_initializer()

sess.run(init_op)

print(sess.run(v1))
print(sess.run(v2))

saver = tf.train.Saver()

checkpoint_path = "./checkpoint/saver2.ckpt"
saver.restore(sess, checkpoint_path)
print("model restored")
print(sess.run(v1))
print(sess.run(v2))

