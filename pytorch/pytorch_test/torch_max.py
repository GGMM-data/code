import torch as tf

a = tf.tensor([1, 2, 3])
b = tf.tensor([2, 1, 4])
print("a: ", a)
print("b: ", b)

c = tf.max(a, b)

print("torch.max(a, b):\n  ", c)
