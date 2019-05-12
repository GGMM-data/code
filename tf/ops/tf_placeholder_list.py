import tensorflow as tf
import numpy as np

n = 4

ph_list = [tf.placeholder(tf.float32, [None, 10]) for _ in range(4)]
result = tf.Variable(0.0)
for x in ph_list:
    result = tf.add(result, x)
hhhh = tf.log(result)


if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    inputs = []
    for _ in range(n):
        x = np.random.rand(16, 10)
        inputs.append(x)
    # print(sess.run(hhhh, feed_dict={ph_list: inputs}))
    print(sess.run(hhhh, feed_dict={k: v for k, v in zip(ph_list, inputs)}).shape)
   
