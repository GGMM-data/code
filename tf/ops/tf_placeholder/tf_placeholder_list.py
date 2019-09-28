import tensorflow as tf
import numpy as np

# 创建一个长度为n的placeholder list
n = 4
ph_list = [tf.placeholder(tf.float32, [None, 10]) for _ in range(4)]
# 对这个ph list的操作
result = tf.Variable(0.0)
for x in ph_list:
    result = tf.add(result, x)
hhhh = tf.log(result)


if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 生成数据
    inputs = []
    for _ in range(n):
        x = np.random.rand(16, 10)
        inputs.append(x)
    # 声明一个字典，存放placeholder和value键值对
    feed_dictionary = {}
    for k,v in zip(ph_list, inputs):
       feed_dictionary[k] = v
    # feed 数据
    print(sess.run(hhhh, feed_dict=feed_dictionary).shape)
   
