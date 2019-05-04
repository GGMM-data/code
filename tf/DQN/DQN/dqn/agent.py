import random
import numpy as np
from tqdm import tqdm
from functools import reduce

import tensorflow as tf

from replaybuffer import ReplayBuffer
from base_model import BaseModel
from history import History
from ops import conv2d, linear, clipped_error


class Agent(BaseModel):
    def __init__(self, config, env, sess):
        super(Agent, self).__init__(config)

        self.sess = sess
        self.env = env
        self.history = History(self.config)
        self.replay_buffer = ReplayBuffer(self.config)

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32',None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()

    def build_dqn(self):
        self.w = {}     # weights
        self.t_w = {}   # target weights

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope('prediction'):
            if self.state_format == 'NHWC':
                self.s_t = tf.placeholder('float32', [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
            else:
                self.s_t = tf.placeholder('float32', [None,  self.history_length, self.screen_height, self.screen_width],
                                          name='s_t')

            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t, 32,[8, 8],[4, 4], initializer=initializer, activation_fn=activation_fn, data_format=self.state_format, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1, 32, [8, 8], [4, 4], initializer=initializer,activation_fn=activation_fn,data_format=self.state_format, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2, 32, [8, 8], [4, 4], initializer=initializer,activation_fn=activation_fn, data_format=self.state_format,
                                                     name='l3')
            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x*y, shape[1:])]) #

            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.l5, self.w['l5_w'], self.w['l5_b'] = linear(self.l4, self.env.action_size, name='q')

            # policy evaluation using max action
            self.q_action = tf.argmax(self.l5, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)   # 对多个batch的q值求平均
            for idx in range(self.env.action_size):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

            # target network
            with tf.variable_scope('target'):
                if self.state_format == 'NHWC':
                    self.target_s_t = tf.placeholder('float32',
                                                     [None, self.screen_height, self.screen_width, self.history_length],
                                                     name='target_s_t')
                else:
                    self.target_s_t = tf.placeholder('float32',
                                                     [None, self.history_length, self.screen_height, self.screen_width],
                                                     name='target_s_t')

                self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t,
                                                                            32, [8, 8], [4, 4], initializer,
                                                                            activation_fn, self.state_format,
                                                                            name='target_l1')
                self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
                                                                            64, [4, 4], [2, 2], initializer,
                                                                            activation_fn, self.state_format,
                                                                            name='target_l2')
                self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
                                                                            64, [3, 3], [1, 1], initializer,
                                                                            activation_fn, self.state_format,
                                                                            name='target_l3')

                shape = self.target_l3.get_shape().as_list()
                self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

                self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                    linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                    linear(self.target_l4, self.env.action_size, name='target_q')

            with tf.variable_scope('pred_to_target'):
                self.t_w_input = {}
                self.t_w_assign_op = {}

                for name in self.t_w.keys():
                    self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                    self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

            with tf.variable_scope('optimizer'):
                self.target_q_t = tf.placeholder('float32', [None], name='target_q_t') # target q at time t

                # self.q_action.eval(s_t)
                self.action = tf.placeholder('int64', [None], name='action')
                action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
                q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

                self.delta = self.target_q_t - q_acted # target q - true q
                self.global_step = tf.Variable(0, trainable=False)

                self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
                self.learning_rate_step = tf.placeholder('int64', None, name='lr_rate_step')
                # self.learning_rate_op = tf.



    def train(self):
        start_step = self.step_op.eval()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen, action, reward, terminal = self.env.new_random_game()

        for _ in range(self.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

