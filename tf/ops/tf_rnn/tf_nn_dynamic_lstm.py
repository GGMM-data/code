import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.nn.rnn_cell as rnn
import time
import numpy as np


def lstm(x, batch_size):
    output_size = 10
    lstm_size = 12  # hidden state and output size

    x = tf.transpose(x, (1, 0, 2))  ## (time_steps, batch_size, state_size)

    lstm = rnn.LSTMCell(lstm_size, forget_bias=1, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32, time_major=True)
    print("hhhhhhhhhhhhhhhhhhhh")
    print(outputs.shape)
    out = tf.convert_to_tensor(outputs[-1:, :, :])
    out = tf.squeeze(out, 0)
    return tf.layers.dense(out, output_size, activation=tf.nn.relu, use_bias=True)


def train():

    mnist = input_data.read_data_sets("~/MNIST_data", one_hot=True)
    print("mnist dataset length:", mnist.train.num_examples)

    train_batch_size = 128
    test_batch_size = 1024
    learning_rate = 0.001
    episodes = 10

    x = tf.placeholder(tf.float32, [None, 28, 28])
    y = tf.placeholder(tf.float32, [None, 10])
    batch_size = tf.placeholder(tf.int32)

    predicted_y = lstm(x, batch_size)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(y, 1))
    # tf.cast
    predict_accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess_conf = tf.ConfigProto()
    sess_conf.gpu_options.allow_growth = True

    with tf.Session(config=sess_conf) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(episodes):
            train_accuracy = 0
            epoches = mnist.train.num_examples // train_batch_size
            begin_time = time.time()
            for j in range(epoches):
                batch_xs, batch_ys = mnist.train.next_batch(train_batch_size)
                batch_xs = np.reshape(np.array(batch_xs), [train_batch_size, 28, 28])
                batch_ys = np.reshape(np.array(batch_ys), [train_batch_size, 10])
                _, accuracy = sess.run([optimizer, predict_accuracy],
                                       feed_dict={x: batch_xs, y: batch_ys, batch_size: train_batch_size})
                train_accuracy += accuracy
            print("Episode %d, train time: %f" % (i, time.time()-begin_time))
            train_accuracy = train_accuracy / epoches
            # print(total_accuracy)

            valid_accuracy = 0
            epoches = mnist.validation.num_examples // test_batch_size
            for j in range(epoches):
                batch_xs, batch_ys = mnist.validation.next_batch(test_batch_size)
                batch_xs = np.reshape(np.array(batch_xs), [test_batch_size, 28, 28])
                batch_ys = np.reshape(np.array(batch_ys), [test_batch_size, 10])
                accuracy = sess.run(predict_accuracy,
                                    feed_dict={x: batch_xs, y: batch_ys, batch_size: test_batch_size})
                valid_accuracy += accuracy
            valid_accuracy = valid_accuracy / epoches
            print("Episodes ", i, ": train accuracy: ", train_accuracy, ", valid_accuracy: ", valid_accuracy)


if __name__ == "__main__":
    train()
