import tensorflow as tf

shape = 3
p_losses_ph = tf.placeholder(tf.float32, shape=[shape], name="ploss")
p_losses_summary = tf.summary.histogram("loss", p_losses_ph)
q_losses_ph = tf.placeholder(tf.float32, shape=[shape], name="qloss")
q_losses_summary = tf.summary.histogram("loss", q_losses_ph)

loss_summary = tf.summary.merge([q_losses_summary, p_losses_summary], name="loss")
writer = tf.summary.FileWriter("../tf_summary/merge")


with tf.Session() as sess:
    inputs1 = [1, 2, 3]
    inputs2 = [1, 2, 3]
    # 错误s_ = sess.run([loss_summary], feed_dict={p_losses_ph: inputs1, q_losses_ph: inputs2})
    # 正确做法1
    s_, = sess.run([loss_summary], feed_dict={p_losses_ph: inputs1, q_losses_ph: inputs2})
    # 正确做法2
    s_ = sess.run(loss_summary, feed_dict={p_losses_ph: inputs1, q_losses_ph: inputs2})
    print(type(s_))
    writer.add_summary(s_, global_step=1)
