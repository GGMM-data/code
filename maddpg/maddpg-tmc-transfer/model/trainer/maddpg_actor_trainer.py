import numpy as np
import tensorflow as tf

import model.common.tf_util as U
from model import AgentTrainer
from model.trainer.replay_buffer import ReplayBuffer
from model.common.ops import q_train, p_act, p_train


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, lstm_model, obs_shape_n, act_space_n, agent_index, actor_env, args, local_q_func=False):
        self.args = args
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
    
        obs_ph_n = []
        for i in range(self.n):
            obs_shape = list(obs_shape_n[i])
            obs_shape.append(args.history_length)
            obs_ph_n.append(U.BatchInput((obs_shape), name="observation" + str(i)).get())
        self.obs_ph_n = obs_ph_n
        self.act_space_n = act_space_n
        self.model = model
        self.lstm_model = lstm_model
        self.local_q_func = local_q_func
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        
        self.p_train, self.p_update = p_train(
            scope=self.name,
            p_scope=actor_env,
            make_obs_ph_n=self.obs_ph_n,
            act_space_n=self.act_space_n,
            p_index=self.agent_index,
            p_func=self.model,
            q_func=self.model,
            lstm_model=self.lstm_model,
            optimizer=self.optimizer,
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units,
            reuse=True,
            use_lstm=False
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(args, obs_shape_n[0], act_space_n[0].n)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None
        
    def update(self, agents, critics, t):
        # 这个就是训练actor的
        # if len(self.replay_buffer) < self.max_replay_buffer_len:
        if len(critics[0].replay_buffer) < self.max_replay_buffer_len:
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        # print("HHHH")
        # collect replay sample from all agents
        obs_n = []  # 长度为n的list，list每隔元素为[batch_size, state_size, history_length]
        obs_next_n = []     # list的每个元素是[bath_size, action_dim]
        act_n = []
        for i in range(self.n):
            # obs, act, rew, obs_next, done = agents[i].replay_buffer.sample()
            obs, act, rew, obs_next, done = critics[i].replay_buffer.sample()
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        # train p network
        p_loss = self.p_train(*(obs_n + act_n + [self.args.batch_size]))

        self.p_update()
        # print("step: ", t, "p_loss: ", p_loss)
        return [p_loss]

    # def update(self, agents, t):
    #     if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
    #         return
    #     if not t % 100 == 0:  # only update every 100 steps
    #         return
    #
    #     self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
    #     # collect replay sample from all agents
    #     obs_n = []
    #     obs_next_n = []
    #     act_n = []
    #     index = self.replay_sample_index
    #     for i in range(self.n):
    #         obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
    #         obs_n.append(obs)
    #         obs_next_n.append(obs_next)
    #         act_n.append(act)
    #     obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
    #
    #     # train q network
    #     num_sample = 1
    #     target_q = 0.0
    #     for i in range(num_sample):
    #         target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
    #         target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
    #         target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
    #     target_q /= num_sample
    #     q_loss = self.q_train(*(obs_n + act_n + [target_q]))
    #
    #     # train p network
    #     p_loss = self.p_train(*(obs_n + act_n))
    #
    #     self.p_update()
    #     self.q_update()
    #     return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

