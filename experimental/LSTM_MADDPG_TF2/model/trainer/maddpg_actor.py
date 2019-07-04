import tensorflow as tf
import os
import sys
cwd = os.getcwd()
path = cwd + "/../"
sys.path.append(path)


import model.common.tf_util as U
from model import AgentTrainer
from model.trainer.replay_buffer import ReplayBuffer
from model.common.ops import p_train, p_act


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, lstm_model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.critic_scope = None
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

        self.obs_ph_n = obs_ph_n
        self.act_space_n = act_space_n
        self.model = model
        self.lstm_model = lstm_model
        self.local_q_func = local_q_func
        self.act, self.p_update, self.p_debug = p_act(
            scope=self.name,
            make_obs_ph_n=self.obs_ph_n,
            act_space_n=self.act_space_n,
            p_index=self.agent_index,
            p_func=self.model,
            lstm_model=self.lstm_model,
            args=self.args,
            reuse=False,
            num_units=self.args.num_units
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        self.p_train = None
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(args, obs_shape_n[0], act_space_n[0].n)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        
    def add_p(self, p_scope):
        self.p_train = p_train(
            scope=self.name,
            make_obs_ph_n=self.obs_ph_n,
            act_space_n=self.act_space_n,
            p_scope=p_scope,
            p_index=self.agent_index,
            p_func=self.model,
            q_func=self.model,
            lstm_model=self.lstm_model,
            optimizer=self.optimizer,
            args=self.args,
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units,
            reuse=True
        )
    
    def change_p(self, p):
        self.p_train = p
        
    def action(self, obs, batch_size):
        return self.act(*(obs + batch_size))

    def experience(self, obs, act, rew, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, float(done), terminal)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        # 这个就是训练actor的
        if len(self.replay_buffer) < self.max_replay_buffer_len:
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
        # train p network
        p_loss = self.p_train(*(obs_n + act_n + [self.args.batch_size]))

        self.p_update()
        # print("step: ", t, "p_loss: ", p_loss)
        return [p_loss]
