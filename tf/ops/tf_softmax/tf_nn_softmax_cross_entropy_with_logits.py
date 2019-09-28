import tensorflow as tf

labels = [[1]]
logits = [[1.0, 1.0, 2.0, 2.0, 2.0, 2.0]]
res_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
sess = tf.Session()
result = sess.run(res_op)
print(result)
print(sum(result))
