import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    def model(x):
      w = tf.Variable([0.3], name="W", dtype=tf.float32)
      B = tf.Variable([-0.3], dtype=tf.float32)
      return w*x + B 

    # input and output
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    predicted_y = model(x) 
    # MSE loss
    loss = tf.reduce_mean(tf.square(y - predicted_y))
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss)

inputs = [1, 2, 3, 4]
outputs = [2, 3, 4, 5]

with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(5000):
        sess.run(train_op, feed_dict={x: inputs, y: outputs})
    l_ = sess.run([loss], feed_dict={x: inputs, y: outputs})
    print("loss: ", l_)
    checkpoint = "./checkpoint/saver3.ckpt"
    save_path = saver.save(sess, checkpoint)
    print("Model has been saved in %s." % save_path)

