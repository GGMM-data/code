import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import data
import numpy as np

inputs = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name="inputs")
filters = tf.get_variable("weigths", [3, 3, 3, 1])
outputs = tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding='SAME')

init_op = tf.global_variables_initializer()

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 1.初始化
        sess.run(init_op)

        # 2.查看所有有可训练的VARIABLES
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var)

        # 3.输入
        img = data.astronaut()
        print("inputs shape: ", img.shape)
        img = np.expand_dims(img, 0)
        print(img.shape)

        # 3.模型预测
        result = sess.run(outputs, feed_dict={inputs: img})

        # 4.查看结果
        print(type(result))
        print(result.shape)

