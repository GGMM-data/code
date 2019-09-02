import numpy as np
import tensorflow as tf

import maddpg_.common.tf_util as U
from maddpg_ import AgentTrainer
from maddpg_.trainer.replay_buffer import ReplayBuffer
from maddpg_.common.ops import q_train, p_act, p_train


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, agents_number, act_space_n, agent_index, actor_scope, args, common_obs_shape, sep_obs_shape,
                model, lstm_model, cnn_model, cnn_scope=None, lstm_scope=None, reuse=False, local_q_func=False, session=None):
        self.args = args
        self.name = name
        self.n = agents_number
        self.agent_index = agent_index
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        
        sep_obs_shape = [args.history_length] + list(sep_obs_shape[1:])
        common_obs_shape = [args.history_length] + list(common_obs_shape)
        common_obs_ph = U.BatchInput(common_obs_shape, name="common_observation").get()
        sep_obs_ph_n = [U.BatchInput(sep_obs_shape, name="common_observation" + str(i)).get() for i in
                        range(self.n)]
        
        self.p_train, self.p_update = p_train(
            scope=self.name,
            p_scope=actor_scope,
            make_common_obs_ph=common_obs_ph,
            make_sep_obs_ph_n=sep_obs_ph_n,
            act_space_n=act_space_n,
            p_index=self.agent_index,
            p_func=model,
            q_func=model,
            cnn_model=cnn_model,
            cnn_scope=cnn_scope,
            lstm_model=lstm_model,
            lstm_scope=lstm_scope,
            optimizer=optimizer,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=self.args.num_units,
            reuse=True,
            use_lstm=self.args.use_lstm,
            session=session,
            args=args
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.history_length)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None
        
    def update(self, agents, critics, t, agent_index):
        # 这个就是训练actor的
        # if len(self.replay_buffer) < self.max_replay_buffer_len:
        if len(critics[0].replay_buffer) < self.max_replay_buffer_len:
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        # print("actor  update")
        self.replay_sample_index = critics[0].replay_buffer.make_index(self.args.batch_size, agent_index)
        # collect replay sample from all agents

        index = self.replay_sample_index
        
        (common_obs_n, sep_obs_n), act_n, rew_n, (common_obs_next_n, sep_obs_next_n), done_n = \
            critics[0].replay_buffer.sample_index(index)
        act, rew, done = act_n[:, agent_index], rew_n[:, agent_index], done_n[:, agent_index]
        # obs, obs_next = sep_obs_n[:, agent_index], sep_obs_next_n[:, agent_index]
        sep_obs_n_list, sep_obs_next_n_list, act_n_list = [], [], []
        for i in range(self.n):
            sep_obs_n_list.append(sep_obs_n[:, :, i])
            sep_obs_next_n_list.append(sep_obs_next_n[:, :, i])
            act_n_list.append(act_n[:, i])
        
        # train p network
        p_loss = self.p_train(*([common_obs_n] + sep_obs_n_list + act_n_list))

        self.p_update()
        # print("step: ", t, "p_loss: ", p_loss)
        return [p_loss]
