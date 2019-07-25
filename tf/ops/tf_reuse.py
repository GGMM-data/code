import tensorflow as tf

# tf.Variable不能用来复用，tf.get_variable()可以。
with tf.variable_scope("v1", reuse=False):
    x = tf.get_variable("x", [2, 2])

with tf.variable_scope("v1", reuse=True):
    x = tf.get_variable("x")
y = x

## 自动复用
with tf.variable_scope("v1", reuse=tf.AUTO_REUSE):
    y1 = tf.get_variable("x", [2, 2])
    y2 = tf.get_variable("y", [2, 4])

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(sess.run(x))
        print(sess.run(y))
        print(sess.run(y1))
        print(sess.run(y2))
        

