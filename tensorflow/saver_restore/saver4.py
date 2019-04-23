import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    def model(x):
        w = tf.Variable([0.3], name="W", dtype=tf.float32)
        B = tf.Variable([-0.3], name="b", dtype=tf.float32)
        predicted_y_ = tf.add(tf.multiply(w, x), B, name="predicted_y")
        return predicted_y_

        # input and output
    x = tf.placeholder(shape=[None], name="x", dtype=tf.float32)
    y = tf.placeholder(shape=[None], name="y", dtype=tf.float32)
    predicted_y = model(x) 
    # MSE loss
    loss = tf.reduce_mean(tf.square(y - predicted_y), name="loss")
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss)

# inputs = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
# outputs = tf.Variable([2, 3, 4, 5], dtype=tf.float32)
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
    checkpoint = "./checkpoint/saver4.ckpt"
    save_path = saver.save(sess, checkpoint)
    print("Model has been saved in %s." % save_path)