import tensorflow as tf

indices = [1, 1, 1]
depth = 5

result = tf.one_hot(indices, depth, on_value=5, off_value=8)

sess = tf.Session()
print(sess.run(result))
