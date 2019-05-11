import numpy as np
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm
from ops import conv2d, linear, clipped_error
from functools import reduce
import gym
from gym import spaces
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


class DQN:
    """Predicted Q value """
    def __init__(self, env, shared_policy, alpha, beta, sess=tf.InteractiveSession(), model_name=""):
        self.sess = sess
        self.env = env
        self.model_name = model_name
        # action space of env
        self.action_dim = self.env.action_space.n
        # pi_0
        self.shared_policy = shared_policy
        # buffer
        self.replay_buffer = deque()
        self.alpha = alpha
        self.beta = beta

        self.replay_buffer_size = REPLAY_BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.episodes = episodes
        self.test_episodes = test_episodes
        self.episode_steps = steps
        self.test_episode_steps = test_steps
        self.gamma = GAMMA

        # create model
        self.initializer = tf.truncated_normal_initializer(0, 0.02)
        self.activation_fn = tf.nn.relu

        self.w = {}
        self.build_model()

    def build_model(self):
        # model layers
        scope = self.model_name + "_" + 'prediction'
        with tf.variable_scope(scope):
            # state

            self.state = tf.placeholder('float32', [None, 84, 84, 3], name='s_t')
            # input action (one hot)
            self.action_one_hot = tf.placeholder("float", [None, self.action_dim])
            self.next_state = tf.placeholder('float32', (None, 84, 84, 3), name='s_t_1')
            self.reward = tf.placeholder('float32', (None,), name='reward')
            self.done = tf.placeholder('int32', (None,), name='done')
            self.times = tf.placeholder('float32', (None,), name='timesteps')

            # cnn layers
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.state, 5, [2, 2], [1, 1], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l1', reuse=False)
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1, 10, [3, 3], [1, 1], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l2', reuse=False)
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2, 10, [3, 3], [1, 1], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l3', reuse=False)
            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            # self.l3_flat = tf.reshape(self.l3, [-1, 200])  #

            # fc layers
            self.q, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, self.action_dim, name='q', reuse=False)

        # optimizer
        with tf.variable_scope(self.model_name + 'optimizer'):
            # predicted q value, action is one hot representation
            self.predicted_q = tf.boolean_mask(self.q, self.action_one_hot)
            # pi0 = self.shared_policy.select_action(self.next_state)
            self.pi0_prob = tf.placeholder(tf.float32, [None, ], name="pi0")
            self.next_q = tf.placeholder(tf.float32, [None, ], name="next_q")
            self.v = tf.log(tf.pow(self.pi0_prob, self.alpha) *
                            tf.exp(self.beta * self.next_q)
                            ) / self.beta

            # true value
            self.y = []
            for i in range(self.batch_size):
                if self.done[i] != 0:
                    self.y.append(self.reward[i])
                else:
                    self.y.append(self.reward[i] + self.gamma * self.v[i])

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
                        tf.train.exponential_decay(self.learning_rate, self.learning_rate_step,
                            self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        tf.global_variables_initializer().run()
        # self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())

    def experience(self, state, action, reward, next_state, done, time):
        # add to model buffer
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done, time))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.popleft()

        # add to  policy buffer
        self.shared_policy.experience(state, action, reward, next_state, done, time)

    def select_action(self, state):
        if len(state.shape) == 3:
            shapes = state.shape
            state = np.reshape(state, (1, ) + shapes)
        # 计算当前状态的Q值
        Q = self.q.eval(feed_dict={self.state: state})
        # pi_0
        # 计算当前状态的action分布
        prob = self.shared_policy.action.eval(feed_dict={self.shared_policy.state: state})
        # 根据公式计算V值，V = tf.pow(pi0, alpha) * tf.exp(beta * Q))
        V = tf.reduce_sum(tf.log((tf.pow(prob, self.alpha) * tf.exp(self.beta * Q))), 1) / self.beta
        # 根据公式2计算pi_i
        pi_i = tf.pow(prob, self.alpha) * tf.exp(self.beta * (Q - V))
        pi_i = pi_i.eval()
        count = tf.ones_like(pi_i)
        hhh = tf.boolean_mask(count, pi_i < 0)
        if tf.greater(tf.reduce_sum(hhh), 0).eval():
            print("Warning!!!: pi_i has negative values: pi_i", pi_i[0])
        pi_i = tf.maximum(tf.zeros(pi_i.shape) + 1e-15, pi_i)
        # 根据pi_i进行采样
        action_sample = tf.multinomial(pi_i, 1).eval()
        return action_sample

    def optimize_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state = [data[0] for data in batch]
        action = [data[1] for data in batch]
        reward = [data[2] for data in batch]
        next_state = [data[3] for data in batch]
        done = [data[4] for data in batch]

        pi_0 = self.select_action(state)
        next_q = self.sess.run(self.q, feed_dict={state: next_state})
        # feed data
        loss, _ = self.sess.run([self.loss, self.optimizer],
                                feed_dict={self.state: state, self.action_one_hot: action, self.reward: reward,
                                           self.next_state: next_state, self.done: done,
                                           self.pi0_prob: pi_0, self.next_q: next_q,
                                           self.learning_rate_step: self.global_step, })
        return loss
