import tensorflow as tf
import numpy as np


# tflearn.layers.conv.conv_2d (
#     incoming, 
#     nb_filter, 
#     filter_size, 
#     strides=1, 
#     padding='same', 
#     activation='linear', 
#     bias=True, 
#     weights_init='uniform_scaling', 
#     bias_init='zeros', 
#     regularizer=None, 
#     weight_decay=0.001, 
#     trainable=True, 
#     restore=True, 
#     reuse=False, 
#     scope=None, 
#     name='Conv2D'
#)

# define graph
inputs = tf.placeholder(tf.float32, [4, 32, 32, 3])
net = tf.layers.conv2d(inputs, filters=32, kernel_size=8, strides=1, activation='relu')
outputs = net


# run
if __name__ == "__main__":
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(var)
    inputs_data = np.ones((4, 32, 32, 3))
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        out = sess.run(outputs, feed_dict={inputs: inputs_data})

    print(out.shape)


