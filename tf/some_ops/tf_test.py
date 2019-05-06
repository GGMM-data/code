import tensorflow as tf

class Model:
  def __init__(self, sess, batch_size):
    self.state = tf.placeholder('float32', (None, ), name='s_t')
    # input action (one hot)
    self.next_state = tf.placeholder('float32', (None, ), name='s_t_1')
    self.reward = tf.placeholder('float32', (None,), name='reward')
    self.done = tf.placeholder('int32', (None,), name='done')
    self.batch_size = batch_size
    self.sess = sess

  def run(self):
    state = tf.random_normal([self.batch_size,], dtype=tf.float32)
    next_state = tf.random_normal([self.batch_size, ], dtype=tf.float32)
    reward = tf.random_normal([self.batch_size, ], dtype=tf.float32)
    done = tf.multinomial(self.batch_size*[[0.5, 0.5]], 1)[0]
    state = self.sess.run(state)
    next_state = self.sess.run(next_state)
    reward = self.sess.run(reward)
    done = self.sess.run(done)
    self.y = []
    for i in range(self.batch_size):
      if self.done[i] == 1:
        self.y.append(self.next_state[i])
      else:
        self.y.append(self.next_state[i] + self.reward[i])

    self.hhh = self.state + self.y 
    print(sess.run(self.hhh, feed_dict={self.state: state, self.next_state: next_state, self.reward: reward, self.done: done}))


sess = tf.Session()
batch_size = 2
model = Model(sess, batch_size)
model.run()
