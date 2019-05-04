import tensorflow as tf

sess = tf.Session()
a = tf.Variable([1, 2, 3])
b = tf.Variable([2, 1, 4])

sess.run(tf.global_variables_initializer())
print("a: ", sess.run(a))
print("b: ", sess.run(b))
c = tf.maximum(a, b)

print("tf.maximum(a, b):\n  ", sess.run(c))
