import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import numpy as np


def lstm(x):
    batch_size = 128
    time_steps = 28
    input_size = 28

    lstm_size = 64
    output_size = 10

    # (batch_size, time_steps, lstm_size)
    # (time_steps, batch_size, input_size)
    x = tf.transpose(x, (1, 0, 2))
    # (time_steps * batch_size, lstm_size)
    x = tf.reshape(x, (-1, input_size))
    # [[batch_size, lstm_size],..., [batch_size, lstm_size]]
    x = tf.split(x, time_steps, 0)
    # 创建一个LSTMCell
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1, state_is_tuple=True)
    zero_state = lstm.zero_state(batch_size, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm, x, initial_state=zero_state, dtype=tf.float32)
    outputs = tf.convert_to_tensor(outputs[-1])
    return tf.layers.dense(outputs, output_size, activation=tf.nn.relu, use_bias=True)


def train():

    mnist = input_data.read_data_sets("/home/mxxmhh/MNIST_data", one_hot=True)

    batch_size = 128
    learning_rate = 0.001
    episodes = 10

    x = tf.placeholder(tf.float32, [None, 28, 28])
    y = tf.placeholder(tf.float32, [None, 10])

    predicted_y = lstm(x)
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(var)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_y, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(y, 1))
    # tf.cast改变Tensor的类型
    predict_accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess_conf = tf.ConfigProto()
    sess_conf.gpu_options.allow_growth = True

    with tf.Session(config=sess_conf) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(episodes):
            train_accuracy = 0
            epoches = mnist.train.num_examples // batch_size
            for j in range(epoches):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs = np.reshape(np.array(batch_xs), [batch_size, 28, 28])
                batch_ys = np.reshape(np.array(batch_ys), [batch_size, 10])
                _, accuracy = sess.run([optimizer, predict_accuracy], feed_dict={x: batch_xs, y:batch_ys})
                train_accuracy += accuracy
            train_accuracy = train_accuracy / epoches
            # print(total_accuracy)

            valid_accuracy = 0
            epoches = mnist.validation.num_examples// batch_size
            for j in range(epoches):
                batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
                batch_xs = np.reshape(np.array(batch_xs), [batch_size, 28, 28])
                batch_ys = np.reshape(np.array(batch_ys), [batch_size, 10])
                accuracy = sess.run(predict_accuracy, feed_dict={x: batch_xs, y: batch_ys})
                valid_accuracy += accuracy
            valid_accuracy = valid_accuracy / epoches
            print("Episodes ", i, ": train accuracy: ", train_accuracy, ", valid_accuracy: ", valid_accuracy)


if __name__ == "__main__":
    train()
