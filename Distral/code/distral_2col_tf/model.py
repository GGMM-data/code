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


class Policy:
    def __init__(self, env, shared_policy, alpha, beta, sess=tf.InteractiveSession()):
        self.sess = sess
        self.env = env
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

    def build_model(self):
        # model layers
        with tf.variable_scope('prediction'):
            # state
            self.state = tf.placeholder('float32', (None, ), name='s_t')

            # cnn layers
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.state, 5, [2, 2], [1, 1], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1, 10, [3, 3], [1, 1], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2, 10, [3, 3], [1, 1], initializer=self.initializer,
                                                             activation_fn=self.activation_fn,
                                                             name='l3')
            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, 200])  #

            # fc layers
            self.q, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, self.action_dim, name='q')

            # output
            self.action = tf.nn.log_softmax(self.q)

        # optimizer
        with tf.variable_scope('optimizer'):
            # input action (one hot)
            self.action_one_hot = tf.placeholder("float", [None, self.action_dim])
            # predicted q value, action is one hot representation
            self.predicted_q = tf.reduce_sum(tf.multiply(self.q, self.action_one_hot),
                                             reduction_indices=1, name='q_acted')
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
                        tf.train.exponential_decay(self.learning_rate, self.learning_rate_step,
                            self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def greedy_action(self, state):
        # q value
        Q = self.q.eval(feed_dict={self.state: [state]})
        # pi_0
        pi_0 = self.shared_policy.action
        # V
        V = tf.log((tf.pow(pi_0, self.alpha) * tf.exp(self.beta * Q)).sum(1)) / self.beta
        # print("pi0 = ", pi0)
        # print(torch.pow(pi0, alpha) * torch.exp(beta * Q))
        # print("V = ", V)
        # use equation 2 to gen pi_i
        pi_i = tf.pow(pi_0, self.alpha) * tf.exp(self.beta * (Q - V))
        if sum(pi_i.data.numpy()[0] < 0) > 0:
            print("Warning!!!: pi_i has negative values: pi_i", pi_i.data.numpy()[0])
        pi_i = tf.max(tf.zeros(pi_i.shape) + 1e-15, pi_i)
        # sample action
        m = Categorical(pi_i)
        action = m.sample().data.view(1, 1)
        return action


    def action(self, state):
        q = self.q.eval(feed_dict={self.state: [state]})
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

        next_state_values = self.q.eval(feed_dict={self.state: next_state})
        y = []

        # calculate true q value
        for i in range(self.batch_size):
            if batch[i][4]:
                y.append(reward[i])
            else:
                next_state_value = np.max(next_state_values[i])
                y.append(reward[i] + self.gamma * next_state_value)
            loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.state: state, self.action_one_hot:action, self.y: y, self.learning_rate_step: self.global_step,})
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
        for i in range(self.test_episodes):
            # print(i)
            state = self.env.reset()
            for step in range(self.test_episode_steps):
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

