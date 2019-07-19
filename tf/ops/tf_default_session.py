import tensorflow as tf


with tf.Session() as sess:
    print("1111111111111111111111111111111")
    print(tf.get_default_session())

sess2 = tf.Session()
print("2222222222222222222222222222222222")
print(tf.get_default_session())

sess3 = tf.Session()
with sess3:
    print("33333333333333333333333333333333")
    print(tf.get_default_session())
