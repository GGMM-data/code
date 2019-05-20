import tensorflow as tf
import numpy as np
from functools import reduce
         
def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(inputs, output_dim, kernel_size, stride, initializer, activation_fn,
           padding='VALID', data_format='NHWC', name="conv2d", reuse=False):
    kernel_shape = None
    with tf.variable_scope(name, reuse=reuse):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], inputs.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], inputs.get_shape()[-1], output_dim ]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(inputs, w, stride, padding, data_format=data_format)

        b = tf.get_variable('b', [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format=data_format)

    if activation_fn is not None:
        out = activation_fn(out)
    return out, w, b


def linear(inputs, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear', reuse=False):
    shape = inputs.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], tf.float32, tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(inputs, w), b)

    if activation_fn != None:
        out = activation_fn(out)
    return out, w, b


class DQN:
    def __init__(self, sess):
        self.sess = sess
        self.w = {}
        self.dim = 10
        self.initializer = tf.truncated_normal_initializer(0, 0.02)
        self.activation_fn = tf.nn.relu
        self.learning_rate = 0.00025

        self.build_model()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        with tf.variable_scope("prediction"):
            self.inputs = tf.placeholder('float32', [None, 84, 84, 3], name='inputs')
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.inputs, 5, [2, 2], [1, 1], initializer=self.initializer, activation_fn=self.activation_fn, name='l1', reuse=False)
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1, 10, [3, 3], [1, 1], initializer=self.initializer, activation_fn=self.activation_fn, name='l2', reuse=False)
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2, 10, [3, 3], [1, 1], initializer=self.initializer, activation_fn=self.activation_fn, name='l3', reuse=False)
            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            # fc layers
            self.q, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, self.dim, name='q', reuse=False)

        with tf.variable_scope('optimizer'):
            # 预测值q
            self.y = tf.placeholder(tf.float32, [None, self.dim], name="y")
            # 目标值(true value)

            # error
            self.delta = self.y - self.q
            # clipped loss function
            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.95, epsilon=0.01).minimize(self.loss)

    def train(self):
        inputs = np.random.rand(16, 84, 84, 3)
        labels = np.random.rand(16, 10)
        for i in range(10):
            _, l = self.sess.run([self.optimizer, self.loss], feed_dict={self.inputs: inputs, self.y: labels})
            print("loss", l)

        

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    dqn = DQN(sess)
    dqn.train()
