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
    def __init__(self, name, agents_number, agent_index, actors, act_space_n, args,
                 common_obs_shape, sep_obs_shape, model, lstm_model, cnn_model, cnn_scope=None, lstm_scope=None,
                 reuse=False, local_q_func=False, session=None):
        self.actors = actors
        self.name = name
        self.n = agents_number
        self.agent_index = agent_index
        self.args = args
        self.history_length = args.history_length
        
        common_obs_shape = [args.history_length] + list(common_obs_shape)
        common_obs_ph = U.BatchInput(common_obs_shape, name="common_observation").get()
        sep_obs_shape = [args.history_length] + list(sep_obs_shape[1:])
        sep_obs_ph_n = [U.BatchInput(sep_obs_shape, name="common_observation" + str(i)).get() for i in
                        range(self.n)]
        
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_common_obs_ph=common_obs_ph,
            make_sep_obs_ph_n=sep_obs_ph_n,
            act_space_n=act_space_n,
            cnn_model=cnn_model,
            cnn_scope=cnn_scope,
            q_index=agent_index,
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
        self.replay_buffer.add(obs, act, rew, new_obs, done)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, agent_index):
         # 训练critic
        # print("hello, nihao a ")
        if len(self.replay_buffer) < self.max_replay_buffer_len:    # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        # print("critic update")
        self.replay_sample_index = agents[0].replay_buffer.make_index(self.args.batch_size, agent_index)
        # collect replay sample from all agents
        index = self.replay_sample_index
        (common_obs_n, sep_obs_n), act_n, rew_n, (common_obs_next_n, sep_obs_next_n), done_n = \
            agents[0].replay_buffer.sample_index(index)
        act, rew, done = act_n[:, agent_index], rew_n[:, agent_index], done_n[:, agent_index]
        # obs, obs_next = sep_obs_n[:, agent_index], sep_obs_next_n[:, agent_index]
        sep_obs_n_list, sep_obs_next_n_list, act_n_list = [], [], []
        for i in range(self.n):
            sep_obs_n_list.append(sep_obs_n[:, :, i])
            sep_obs_next_n_list.append(sep_obs_next_n[:, :, i])
            act_n_list.append(act_n[:, i])
        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            # obs_next_n[j]是第j个智能体的observation
            # p_debug的输入应该是[[?, 4, 30, 30, 3], [?, 4, 4]]
            target_act_next_n = [self.actors[j].p_debug['target_act'](common_obs_n, sep_obs_next_n[:, :, j]) for j in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*([common_obs_next_n] + sep_obs_next_n_list + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        q_loss = self.q_train(*([common_obs_n] + sep_obs_n_list + act_n_list + [target_q]))
        # train p network

        self.q_update()
        # print("step: ", t, "q_loss: ", q_loss)33
        return [q_loss, np.mean(target_q), np.mean(rew), np.std(target_q)]
