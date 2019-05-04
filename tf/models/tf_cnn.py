import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/home/mxxmhh/MNIST_data", one_hot=True)

batch_size = 128

sess_conf = tf.ConfigProto()
sess_conf.gpu_options.allow_growth = True

inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="inputs")
weights = tf.Variable(tf.truncated_normal([4, 4, 1, 3], 0, 0.1))
outputs = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
outputs = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, 1, 1, 1], padding='VALID')

with tf.Session(config=sess_conf) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        train_accuracy = 0
        epoches = mnist.train.num_examples // batch_size
        for j in range(epoches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(np.array(batch_xs), [batch_size, 28, 28, 1])
            batch_ys = np.reshape(np.array(batch_ys), [batch_size, 10])
            print(type(batch_xs))
            output = sess.run(outputs, feed_dict={inputs: batch_xs})
            print(output.shape)

