import tensorflow as tf

# tf.multinomial(logits, num_samples, seed=None, name=None)
# logits 是一个二维张量，指定概率，num_samples是采样个数
sess = tf.Session()
sample = tf.multinomial([[5.0, 5.0, 5.0], [5.0, 4, 3]], 10) # 注意logits必须是float
for _ in range(5):
  print(sess.run(sample))
  
