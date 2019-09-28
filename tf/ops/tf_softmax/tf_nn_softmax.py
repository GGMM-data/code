import tensorflow as tf
import numpy as np

logits = [[1.0, 1.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]
res_op = tf.nn.softmax(logits)
sess = tf.Session()
result = sess.run(res_op)
print(result)
print(np.sum(result, 1))
