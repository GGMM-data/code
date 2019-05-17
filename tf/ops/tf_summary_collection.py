import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    # model parameters
    w = tf.Variable([0.3], name="w", dtype=tf.float32)
    b = tf.Variable([0.2], name="b", dtype=tf.float32)

    x = tf.placeholder(tf.float32, name="inputs")
    y = tf.placeholder(tf.float32, name="outputs")

    with tf.name_scope('linear_model'):
        linear = w * x + b

    with tf.name_scope('cal_loss'):
        loss = tf.reduce_mean(input_tensor=tf.square(y - linear), name='loss')

    with tf.name_scope('add_summary'):
        summary_loss = tf.summary.scalar('MSE', loss)
        summary_b = tf.summary.scalar('b', b[0])
        summary_w = tf.summary.scalar('w', w[0])

    with tf.name_scope('train_model'):
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

inputs = [1, 2, 3, 4]
outputs = [2, 3, 4, 5]

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./summary/", graph)
    # merged = tf.summary.merge_all()

    for i in range(100):
        #_, summ = sess.run([train, merged], feed_dict={x: inputs, y: outputs})
        #writer.add_summary(summ, global_step=i)
        _, sl, sw, sb = sess.run([train, summary_loss, summary_w, summary_b], feed_dict={x: inputs, y: outputs})
        writer.add_summary(sl, global_step=i)
        writer.add_summary(sw, global_step=i)
        writer.add_summary(sb, global_step=i)

    w_, b_, l_ = sess.run([w, b, loss], feed_dict={x: inputs, y: outputs})
    print("w: ", w_, "b: ", b_, "loss: ", l_)
    for var in tf.get_collection(tf.GraphKeys.SUMMARIES):
        print(var)
    writer.close()
