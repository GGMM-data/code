import tensorflow as tf
import matplotlib.pyplot as plt
n = 32
x = tf.linspace(-3.0, 3.0, n)

# 1.创建一个session
sess = tf.Session()

## 在sess内执行运行
## 方法1
result  = sess.run(x)
## 方法2
# x.eval()
x.eval(session=sess)
sess.close()

# 2.创建一个交互式sess
sess = tf.InteractiveSession()
x.eval()

# 3.新的操作被添加到默认图上
sigma = 1.0
mean = 0.0
# 和x的shape是一样的
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                        (2.0 * tf.pow(sigma, 2.0)))) *
     (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
print(type(z))
print(z.graph is tf.get_default_graph())

plt.plot(z.eval())
plt.show()

# 4. 查看shape
print(z.shape)
print(z.get_shape())
print(z.get_shape().as_list())
print(tf.shape(z).eval())

# 5.some function
# tf.stack
print(tf.stack([tf.shape(z),tf.shape(z),[3]]).eval())
# tf.reshape, tf.matmul
z_ = tf.matmul(tf.reshape(z, (n, 1)), tf.reshape(z, (1, n)))
plt.imshow(z_.eval()) plt.show()

# tf.ones_like, tf.multiply
# tf.ones_like返回与输入tensor具有相同shape的tensor
x = tf.reshape(tf.sin(tf.linspace(- 3.0, 3.0, n)), (n, 1))
print(x.shape)
y = tf.reshape(tf.ones_like(x), (1, n))
print(y.shape)
print(y.eval())
z = tf.multiply(tf.matmul(x,y), z_)
print(z.shape)
plt.imshow(z.eval())
plt.show()

# 6.列出graph中的所有操作
ops = tf.get_default_graph().get_operations()
print([op for op in ops])

# 7.卷积conv
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
