import tensorflow as tf

x1 = tf.constant(1.0)
l1 = tf.nn.l2_loss(x1)

x2 = tf.constant([2.5, -0.3])
l2 = tf.nn.l2_loss(x2)

tf.add_to_collection("losses", l1)
tf.add_to_collection("losses", l2)

losses = tf.get_collection('losses')
for var in tf.get_collection('losses'):
    print(var)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    losses_val = sess.run(losses)
    print(losses_val)
