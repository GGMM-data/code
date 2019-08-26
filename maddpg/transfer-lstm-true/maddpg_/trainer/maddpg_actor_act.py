import os
import sys
import numpy as np

sys.path.append(os.getcwd() + "/../")

import maddpg_.common.tf_util as U
from maddpg_ import AgentTrainer
from maddpg_.trainer.replay_buffer import ReplayBuffer
from maddpg_.common.ops import p_act


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, agents_number, act_space_n, agent_index, args, common_obs_shape, sep_obs_shape,
                 model, lstm_model, cnn_model, lstm_scope=None, cnn_scope=None, reuse=False, session=None, local_q_func=False):
        self.args = args
        self.name = name
        self.n = agents_number
        self.agent_index = agent_index
        self.local_q_func = local_q_func
        
        sep_obs_shape = [args.history_length] + list(sep_obs_shape[1:])
        common_obs_shape = [args.history_length] + list(common_obs_shape)
   
        common_obs_ph = U.BatchInput(common_obs_shape, name="common_observation").get()
        sep_obs_ph_n = [U.BatchInput(sep_obs_shape, name="common_observation"+str(i)).get() for i in range(self.n)]
        
        self.act, self.p_debug = p_act(
            make_common_obs_ph=common_obs_ph,
            make_sep_obs_ph_n=sep_obs_ph_n,
            act_space_n=act_space_n,
            p_index=self.agent_index,
            p_func=model,
            lstm_model=lstm_model,
            cnn_model=cnn_model,
            lstm_scope=lstm_scope,
            cnn_scope=cnn_scope,
            use_lstm=self.args.use_lstm,
            use_cnn=self.args.use_cnn,
            reuse=reuse,
            session=session,
            scope=self.name,
            num_units=self.args.num_units,
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.history_length)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
  
    def action(self, common_obs, sep_obs):
        # obs = np.array(obs.queue)
        # print(obs)
        return self.act(common_obs[None], sep_obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))


