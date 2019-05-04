import numpy as np
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm
from ops import conv2d, linear, clipped_error
from functools import reduce
import gym
import time

scale = 10

ENV_NAME = "Breakout-v4"
GAMMA = 0.9 
REPLAY_BUFFER_SIZE = 100 * scale
BATCH_SIZE = 32
EPSILON = 0.01
HIDDEN_UNITS = 512

learning_rate = 0.00025
learning_rate_minimum = 0.00025
learning_rate_decay = 0.96
learning_rate_decay_step = 5 * scale

episodes = 3000
test_episodes = 10
steps = 300
test_steps = 300

ep_end = 0.1
ep_start = 1.
ep_end_t = REPLAY_BUFFER_SIZE


class DQN:
    def __init__(self, env, env_name, sess=tf.InteractiveSession()):
        self.env = env
        self.env_name = env_name
        self.model_dir = self.env_name + "/"
        self.sess = sess
        self.replay_buffer = deque()
        self.state_dim = env.observation_space.shape
        self.height = 84
        self.width = 84
        self.action_dim = env.action_space.n
        self.hidden_dim = HIDDEN_UNITS

        self.ep_start = ep_start
        self.ep_end = ep_end
        self.ep_end_t = ep_end_t

        self.episodes = episodes
        self.episode_steps = steps

        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.replay_buffer_size = REPLAY_BUFFER_SIZE
        self.epsilon = EPSILON

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        # create model
        self.initializer = tf.truncated_normal_initializer(0, 0.02)
        self.activation_fn = tf.nn.relu

        self.w = {}
        # model layers
        with tf.variable_scope('prediction'):
            # state
            self.state_input = tf.placeholder('float32', (None, ) + self.state_dim, name='s_t')
            self.state = tf.image.resize_images(self.state_input, [84, 64])
            self.state = tf.image.pad_to_bounding_box(self.state, 0, 10, self.height, self.width)

            # cnn layers
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.state, 32, [8, 8], [4, 4], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1, 32, [4, 4], [2, 2], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2, 32, [3, 3], [1, 1], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l3')
            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])  #

            # fc layers
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, self.hidden_dim, activation_fn=self.activation_fn, name='l4')
            self.q, self.w['l5_w'], self.w['l5_b'] = linear(self.l4, self.action_dim, name='q')

            # policy evaluation using max action
            self.q_action = tf.argmax(self.q, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)  # 对多个batch的q值求平均
            for idx in range(self.action_dim):
              q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        # optimizer
        with tf.variable_scope('optimizer'):
            # input action (one hot)
            self.action_one_hot = tf.placeholder("float", [None, self.action_dim])
            ### input action (not one hot)
            # self.action_not_one_hot = tf.placeholder('int64', [None], name='action')
            ### action one hot
            # self.action_one_hot = tf.one_hot(self.action_not_one_hot, self.env.action_size, 1.0, 0.0, name='action_one_hot')

            # predicted q value, action is one hot representation
            self.predicted_q = tf.reduce_sum(tf.multiply(self.q, self.action_one_hot), reduction_indices=1, name='q_acted')
            # true value
            self.y = tf.placeholder("float", [None])
            # error
            self.delta = self.y - self.predicted_q
            # clipped loss function
            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')

            self.global_step = tf.Variable(0, trainable=False)

            self.learning_rate = learning_rate
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_decay_step = learning_rate_decay_step
            self.learning_rate_decay = learning_rate_decay
            self.learning_rate_minimum = learning_rate_minimum

            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                tf.train.exponential_decay(
                    self.learning_rate,
                    self.learning_rate_step,
                    self.learning_rate_decay_step,
                    self.learning_rate_decay,
                    staircase=True))
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)


        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
              self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
              self.summary_ops[tag]  = tf.summary.scalar("%s/%s" % (self.env_name, tag), self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
              self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
              self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

            self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self._saver = tf.train.Saver(list(self.w.values()) + [self.step_op], max_to_keep=30)
        # l = list(self.w.values())
        # print(l)

    def greedy_action(self, state):
        ep = self.ep_end + max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - self.global_step) / self.ep_end_t)
        if random.random() < ep:
            return np.random.randint(0, self.action_dim - 1)
        else:
            q = self.q.eval(feed_dict={self.state_input: [state]})
            return np.argmax(q)

    def action(self, state):
        q = self.q.eval(feed_dict={self.state_input: [state]})
        return np.argmax(q)

    def experience(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > self.replay_buffer_size:
           self.replay_buffer.popleft()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
           return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state = [data[0] for data in batch]
        action = [data[1] for data in batch]
        reward = [data[2] for data in batch]
        next_state = [data[3] for data in batch]

        next_state_values = self.q.eval(feed_dict={self.state_input: next_state})
        y = []

        # calculate true q value
        for i in range(self.batch_size):
          if batch[i][4]:
            y.append(reward[i])
          else:
            next_state_value = np.max(next_state_values[i])
            y.append(reward[i] + self.gamma * next_state_value)
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.state_input: state, self.action_one_hot:action, self.y: y, self.learning_rate_step: self.global_step,})
        return loss

    def train(self):
        # begin to train
        self.global_step = 0
        for i in range(self.episodes):
            # reset environment
            state = self.env.reset()
            for step in range(self.episode_steps):
                self.global_step += 1
                # choose action
                action = self.greedy_action(state)
                # run a step
                next_state, reward, done, _ = self.env.step(action)
                # reset reward
                reward = -1 if done else 0.1
                # add transition to replay buffer
                self.experience(state, action, reward, next_state, done)

                # update parameters
                loss = self.update()
                if step % 100 == 0:
                    print("Train episode ", i, ", step ", step, ", loss: ", loss)
                # move to next state
                state = next_state
                if done:
                    break
            # after train one episode, test
            self.test()

    def test(self):
        total_reward = 0
        for i in range(test_episodes):
            # print(i)
            state = self.env.reset()
            for step in range(test_steps):
                # env.render()
                action = self.action(state)
                # print(action)
                next_state, reward, done, _ = self.env.step(action)
                if reward != 0.0:
                    print(reward)
                total_reward += reward
                if done:
                    break
        average_reward = total_reward / test_episodes
        print("Average reward test episode: ", average_reward)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    model = DQN(env=env,env_name=ENV_NAME)
    model.train()
