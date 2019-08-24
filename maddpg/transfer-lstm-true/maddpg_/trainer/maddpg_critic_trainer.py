import numpy as np
import tensorflow as tf
import os
import sys
cwd = os.getcwd()
path = cwd + "/../"
sys.path.append(path)

import maddpg_.common.tf_util as U
from maddpg_ import AgentTrainer
from maddpg_.trainer.replay_buffer import ReplayBuffer
from maddpg_.common.ops import q_train


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, lstm_model, cnn_model, obs_shape_n, act_space_n, agent_index, actors, args, local_q_func=False,
                 session=None, lstm_scope=None):
        self.actors = actors
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.history_length = args.history_length
        
        obs_ph_n = []
        for i in range(self.n):
            obs_shape = [args.history_length] + list(obs_shape_n[i])
            obs_ph_n.append(U.BatchInput((obs_shape), name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            cnn_model=cnn_model,
            q_func=model,
            lstm_model=lstm_model,
            lstm_scope=lstm_scope,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            args=self.args,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            reuse=False,
            use_lstm=self.args.use_lstm,
            session=session
        )

        self.replay_buffer = ReplayBuffer(args.buffer_size, args.history_length)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        # 训练critic
        # print("hello, nihao a ")
        if len(self.replay_buffer) < self.max_replay_buffer_len:    # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        # print("critic update")
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            # buffer
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        
        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [self.actors[j].p_debug['target_act'](obs_next_n[i]) for j in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        q_loss = self.q_train(*(obs_n + act_n + [target_q]))
        # train p network

        self.q_update()
        # print("step: ", t, "q_loss: ", q_loss)
        return [q_loss, np.mean(target_q), np.mean(rew), np.std(target_q)]
