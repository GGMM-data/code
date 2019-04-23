import tensorflow as tf

# 你所定义的变量一定要在 checkpoint 中存在；但不是所有在checkpoint中的变量，你都要重新定义。你恢复的变量也必须有定义

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)

# input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
predicted_y = W * x + b
loss = tf.reduce_mean(tf.square(y - predicted_y))

inputs = [1, 2, 3, 4]
outputs = [2, 3, 4, 5]

with tf.Session() as sess:
    saver = tf.train.Saver()
    checkpoint = "./checkpoint/saver1.ckpt"
    saver.restore(sess, checkpoint)
    l_, W_, b_ = sess.run([loss, W, b], feed_dict={x: inputs, y: outputs})
    print("loss: ", l_, "w: ", W_, "b:", b_)
    print("Model has been restored.")
