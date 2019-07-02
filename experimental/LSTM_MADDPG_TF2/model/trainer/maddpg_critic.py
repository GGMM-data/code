import numpy as np
import tensorflow as tf
import os
import sys
cwd = os.getcwd()
path = cwd + "/../"
sys.path.append(path)

import model.common.tf_util as U
from model import AgentTrainer
from model.trainer.replay_buffer import ReplayBuffer
from model.common.ops import q_train, p_act


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, lstm_model, obs_shape_n, act_space_n, agent_index, shared_actor, args, local_q_func=False):
        # 最终的目标actor
        self.shared_actor = shared_actor
        
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []

        for i in range(self.n):
            obs_shape = list(obs_shape_n[i])
            obs_shape.append(args.history_length)
            obs_ph_n.append(U.BatchInput((obs_shape), name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            lstm_model=lstm_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            args=self.args,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            reuse=False
        )

        self.replay_buffer = ReplayBuffer(args, obs_shape_n[0], act_space_n[0].n)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.p = None

    def experience(self, obs, act, rew, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, float(done), terminal)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        # 训练critic
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        # collect replay sample from all agents
        obs_n = []  # 长度为n的list，list每隔元素为[batch_size, state_size, history_length]
        obs_next_n = []     # list的每个元素是[bath_size, action_dim]
        act_n = []
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample()
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample()

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [self.shared_actor[i].p_debug['target_act'](*([obs_next_n[i]] + [self.args.batch_size])) for j in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n + [self.args.batch_size]))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        q_loss = self.q_train(*(obs_n + act_n + [target_q] + [self.args.batch_size]))
        # train p network

        self.q_update()
        # print("step: ", t, "q_loss: ", q_loss)
        return [q_loss, np.mean(target_q), np.mean(rew), np.std(target_q)]
