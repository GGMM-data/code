import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import data
import numpy as np

def conv(img):
    if len(img.shape) == 3:
        img = tf.reshape(img, [1]+img.get_shape().as_list())
    fiter = tf.random_normal([3, 3, 3, 1])
    print(type(filter))
    img = tf.nn.conv2d(img, fiter, strides=[1, 1, 1, 1], padding='SAME')
    print(img.get_shape())
    return img

init_op = tf.global_variables_initializer()
if __name__ == "__main__":
    # img = data.text()
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    img = data.astronaut()
    print(img.shape)
    plt.imshow(img)
    plt.show()
    
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    # 输出为空

    x = tf.placeholder(tf.float32, shape=(img.shape))
    y = conv(x)
    result = sess.run(y, feed_dict={x: img})
    print(type(result))
    result = np.squeeze(result) 
    plt.imshow(result)
    plt.show()
