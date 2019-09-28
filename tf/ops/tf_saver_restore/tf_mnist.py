import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def build_cnn(images, labels):
    conv1 = tf.layers.conv2d(inputs=images, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # 生成prediction，一个是类别[batch_size, 1]，一个是概率
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    labels = tf.argmax(labels, 1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    accuracy = tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
    return predictions, loss, train_op, accuracy


if __name__ == "__main__":
    mnist = input_data.read_data_sets("/home/mxxmhh/MNIST_data", one_hot=True)
    batch_size = 128
    images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="images")
    labels = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    sess_conf = tf.ConfigProto()
    sess_conf.gpu_options.allow_growth = True

    ckpt_path = "./ckpt/model.ckpt"
    with tf.Session(config=sess_conf) as sess:
        predictions, loss, train_op, accuracy = build_cnn(images, labels)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print(tf.get_default_graph())
        saver = tf.train.Saver()
        for step in range(10):
            epoches = mnist.train.num_examples // batch_size
            for j in range(epoches):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs = np.reshape(np.array(batch_xs), [batch_size, 28, 28, 1])
                batch_ys = np.reshape(np.array(batch_ys), [batch_size, 10])
                _, acc = sess.run([train_op, accuracy], feed_dict={images: batch_xs, labels: batch_ys})
            print(acc)
            saver.save(sess, ckpt_path, step)
    
