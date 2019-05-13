import tensorflow as tf

# 函数：定义一个summary标量
# tf.summary.scalar(name, tensor, collections=None, family=None)

# 类：将数据写入文件
# tf.summary.FileWriter(self, logdir,　graph=None, max_queue=10,flush_secs=120, graph_def=None, filename_suffix=None)
# 类内函数：将summary类型变量转换为事件
# tf.summary.FileWriter.add_summary(self, summary, global_step=None)
# 
# 使用tensorboard --logdir ./summary/打开tensorboard

# summary_loss = tf.summary.scalar('loss', loss)
# summary_weights = tf.summary.scalar('weights', weights)
# writer = tf.summary.FileWriter("./summary/")
# sess = tf.Session()
# loss_, weights_ = sess.run([summary_loss, summary_weights], feed_dict={})
# writer.add_summary(loss_)
# writer.add_summary(weights_)
# 或者
# 先把loss和weights merge 一下，然后再run，这样子就不用写那么多op了
# merged = tf.summary.merge_all() 
# merged_ = sess.rum([merged], feed_dict={}) # 这里的merged相当于summary_weight和summary_loss
# writer.add_summary(merged_, global_step)

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

    with tf.name_scope('train_model'):
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

inputs = [1, 2, 3, 4]
outputs = [2, 3, 4, 5]

with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter("./summary/", graph)
    merged = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(5000):
        _, summ = sess.run([train, merged], feed_dict={x: inputs, y: outputs})
        writer.add_summary(summ, global_step=i)

    w_, b_, l_ = sess.run([w, b, loss], feed_dict={x: inputs, y: outputs})
    print("w: ", w_, "b: ", b_, "loss: ", l_)
    for var in tf.get_collection(tf.GraphKeys.SUMMARIES):
    #for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
        print(var)

