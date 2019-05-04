import tensorflow as tf

# x1 = tf.Variable(tf.constant([3,4]))
x1 = tf.Variable([3,4])
# x2 = tf.Variable(tf.constant([9,1]))
x2 = tf.Variable([9,1])

y = x1.assign(x2)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #sess.run(x1.initializer)
  xx1 = sess.run(x1)
  print(xx1)

  xx2 = sess.run(x2)
  print(xx2)

  print(sess.run(x1))
  yy = sess.run(y)
  print(yy)
  print(sess.run(x1))
  print(sess.run(x2))
