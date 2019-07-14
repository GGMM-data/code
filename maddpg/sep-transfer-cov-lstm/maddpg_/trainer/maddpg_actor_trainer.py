import numpy as np
import tensorflow as tf

import maddpg_.common.tf_util as U
from maddpg_ import AgentTrainer
from maddpg_.trainer.replay_buffer import ReplayBuffer
from maddpg_.common.ops import q_train, p_act, p_train


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, lstm_model, obs_shape_n, act_space_n, agent_index, actor_env, args, local_q_func=False):
        self.args = args
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
    
        obs_ph_n = []
        for i in range(self.n):
            obs_shape = [args.history_length] + list(obs_shape_n[i])
            # obs_shape.append()
            obs_ph_n.append(U.BatchInput((obs_shape), name="observation"+str(i)).get())
            
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        
        self.p_train, self.p_update = p_train(
            scope=self.name,
            p_scope=actor_env,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=self.agent_index,
            p_func=model,
            q_func=model,
            lstm_model=lstm_model,
            optimizer=optimizer,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=self.args.num_units,
            reuse=True,
            use_lstm=True
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
        
    def update(self, agents, critics, t, index):
        # 这个就是训练actor的
        # if len(self.replay_buffer) < self.max_replay_buffer_len:
        if len(critics[0].replay_buffer) < self.max_replay_buffer_len:
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        
        self.replay_sample_index = critics[index].replay_sample_index
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            # buffer
            obs, act, rew, obs_next, done = critics[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        # print("step: ", t, "p_loss: ", p_loss)
        return [p_loss]
