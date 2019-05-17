import tensorflow as tf

# 你所定义的变量一定要在 checkpoint 中存在；但不是所有在checkpoint中的变量，你都要重新定义。你恢复的变量也必须有定义
inputs = [1.0, 2.0, 3.0, 4.0]
outputs = [2.0, 3.0, 4.0, 5.0]

with tf.Session() as sess:
    checkpoint = "./checkpoint/saver4.ckpt"
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.import_meta_graph(checkpoint+".meta")
    # saver.restore(sess, tf.train.latest_checkpoint("./checkpoint/"))
    saver.restore(sess, checkpoint)
    graph = tf.get_default_graph()
    loss = graph.get_tensor_by_name('loss:0')
    w = graph.get_tensor_by_name('W:0')
    b = graph.get_tensor_by_name("b:0")
    predicted_y = graph.get_tensor_by_name("predicted_y:0")
    print(sess.run(w))
    print(sess.run(b))
    x = tf.placeholder(shape=[None], name="x", dtype=tf.float32)
    y = tf.placeholder(shape=[None], name="y", dtype=tf.float32)
    print(sess.run(predicted_y, feed_dict={x: inputs, y: outputs}))

    # sess.run(loss, feed_dict={x: inputs, y: outputs})
    sess.run(loss, feed_dict={x: 1.0, y: [[2.0]]})
    print()
    print("Model has been restored.")

# def model(x):
#   W = tf.Variable([0.3],name="W", dtype=tf.float32)
#   b = tf.Variable([-0.3], dtype=tf.float32)
#   return W * x + b
#
# # input and output

# predicted_y = model(x)
# loss = tf.reduce_mean(tf.square(y - predicted_y))
#
