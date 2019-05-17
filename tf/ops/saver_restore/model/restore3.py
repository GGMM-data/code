import tensorflow as tf

# 你所定义的变量一定要在 checkpoint 中存在；但不是所有在checkpoint中的变量，你都要重新定义。你恢复的变量也必须有定义
def model(x):
  W = tf.Variable([0.3],name="W", dtype=tf.float32)
  b = tf.Variable([-0.3], dtype=tf.float32)
  return W * x + b

# input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
predicted_y = model(x) 
loss = tf.reduce_mean(tf.square(y - predicted_y))

inputs = [1, 2, 3, 4]
outputs = [2, 3, 4, 5]

with tf.Session() as sess:
    saver = tf.train.Saver()
    checkpoint = "./checkpoint/saver3.ckpt"
    saver.restore(sess, checkpoint)
    l_ = sess.run([loss], feed_dict={x: inputs, y: outputs})
    print("loss: ", l_)
    print("Model has been restored.")
