import os
import sys
import numpy as np

sys.path.append(os.getcwd() + "/../")

import maddpg_.common.tf_util as U
from maddpg_ import AgentTrainer
from maddpg_.trainer.replay_buffer import ReplayBuffer
from maddpg_.common.ops import p_train, p_act


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, lstm_model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False, session=None):
        self.args = args
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        
        obs_ph_n = []
        for i in range(self.n):
            obs_shape = [args.history_length] + list(obs_shape_n[i])
            obs_ph_n.append(U.BatchInput((obs_shape), name="observation"+str(i)).get())
        
        self.local_q_func = local_q_func
        self.act, self.p_debug = p_act(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=self.agent_index,
            p_func=model,
            lstm_model=lstm_model,
            num_units=self.args.num_units,
            use_lstm=self.args.use_lstm,
            reuse=False,
            session=session
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.history_length)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
  
    def action(self, obs):
        obs = np.array(obs.queue)
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))


