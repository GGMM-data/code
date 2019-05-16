import tensorflow as tf


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

# tf.train.RMSPropOptimizer()







