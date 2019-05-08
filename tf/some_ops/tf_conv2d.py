import tensorflow as tf

def conv(img):
    if len(img.shape) == 3:
        img = tf.reshape(img, [1]+img.get_shape().as_list())
    fiter = tf.random_normal([3, 3, 3, 1])
    img = tf.nn.conv2d(img, fiter, strides=[1, 1, 1, 1], padding='SAME')
    print(img.get_shape())
    return img

from skimage import data
# img = data.text()
img = data.astronaut()
print(img.shape)
plt.imshow(img)
plt.show()

x = tf.placeholder(tf.float32, shape=(img.shape))
result = tf.squeeze(conv(x)).eval(feed_dict={x:img})
plt.imshow(result)
plt.show()
