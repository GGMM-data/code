import tensorflow as tf


print(tf.global_variables())
a = tf.get_variable("a", shape=[3, 3])

print(tf.global_variables())
init_op = tf.global_variables_initializer()
# 就是调用了variables_initializer(global_variables())

sess = tf.Session()
sess.run(init_op)
print(sess.run(a))
