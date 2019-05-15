import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal_1000/moving_mean_50", mean_moving_normal)

# mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# tf.summary.histogram("normal_1000/moving_mean_10", mean_moving_normal)
# mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# tf.summary.histogram("normal_1000", mean_moving_normal)

# mean_moving_normal = tf.random_normal(shape=[100], mean=(5*k), stddev=1)
# tf.summary.histogram("normal_100/moving_mean_10", mean_moving_normal)
# mean_moving_normal = tf.random_normal(shape=[100], mean=(5*k), stddev=1)
# tf.summary.histogram("normal_100", mean_moving_normal)

# mean_moving_normal = tf.random_normal(shape=[10], mean=(5*k), stddev=1)
# tf.summary.histogram("normal_10/moving_mean_10", mean_moving_normal)
# mean_moving_normal = tf.random_normal(shape=[10], mean=(5*k), stddev=1)
# tf.summary.histogram("normal_10", mean_moving_normal)

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("summary/histogram/")

summaries = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
N = 50
for step in range(N):
  print(step)
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step)

writer.close()
