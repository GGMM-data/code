import tensorflow as tf
import os
import sys

sys.path.append(os.getcwd() + "/../")
import h5py

from maddpg_.trainer.maddpg_actor_trainer import MADDPGAgentTrainer as ACTOR_TRAINER
from maddpg_.trainer.maddpg_critic_trainer import MADDPGAgentTrainer as CRITIC_TRAINER
from maddpg_.trainer.maddpg_actor_act import MADDPGAgentTrainer as ACT
import tensorflow.contrib.layers as layers
import tensorflow.nn.rnn_cell as rnn
import time
import numpy as np
from multiagent.uav.flag import FLAGS


def sample_map(path, random=False):
    f = h5py.File(path, "r")
    data = f['data'][:]
    f.close()
    data_shape = data.shape
    map_size = FLAGS.size_map
    index = np.random.randint(0, data_shape[0])
    if random:
        map_beigin = np.random.randint(0, data_shape[0] - map_size)
        map = np.sum(data[index, map_beigin:map_beigin+map_size, map_beigin:map_beigin+map_size], 2)
    else:
        map = np.sum(data[index, :map_size, :map_size], 2)
    return map


def make_env(scenario_name, benchmark=False, reward_type=0):
    from multiagent.environment_uav import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            scenario.benchmark_data, reward_type=reward_type)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, reward_type=reward_type)
    return env


def get_trainers(env, env_name, num_adversaries, arglist, common_obs_shape, sep_obs_shape,
        lstm_scope=None, cnn_scope=None, reuse=False, session=None, actors=None, actor_scope=None,  agent_type=0):
    
    trainers = []
    model = mlp_model
    lstm = lstm_model
    cnn = cnn_model
    if agent_type == 0:
        trainer = ACT
        for i in range(num_adversaries):
            trainers.append(trainer(
                name=env_name + "agent_%d" % i, agents_number=env.n,
                common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape, act_space_n=env.action_space,
                model=model, lstm_model=lstm, cnn_model=cnn, lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                reuse=reuse, session=session, local_q_func=(arglist.adv_policy == 'ddpg'), agent_index=i, args=arglist))
            
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                name=env_name + "agent_%d" % i, agents_number=env.n,
                common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape, act_space_n=env.action_space,
                model=model, lstm_model=lstm, cnn_model=cnn, lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                reuse=reuse, session=session, local_q_func=(arglist.adv_policy == 'ddpg'), agent_index=i, args=arglist))
            
    elif agent_type == 1:
        trainer = CRITIC_TRAINER
        for i in range(num_adversaries):
            trainers.append(trainer(
                actors=actors,
                name=env_name + "agent_%d" % i, agents_number=env.n,
                common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape, act_space_n=env.action_space,
                model=model, lstm_model=lstm, cnn_model=cnn, lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                reuse=reuse, session=session, agent_index=i, local_q_func=(arglist.adv_policy == 'ddpg'), args=arglist))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                actors=actors,
                name=env_name + "agent_%d" % i, agents_number=env.n,
                common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape, act_space_n=env.action_space,
                model=model, lstm_model=lstm, cnn_model=cnn, lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                reuse=reuse, session=session, agent_index=i, local_q_func=(arglist.adv_policy == 'ddpg'), args=arglist))
    elif agent_type == 2:
        trainer = ACTOR_TRAINER
        for i in range(num_adversaries):
            trainers.append(trainer(
                actor_scope=actor_scope + "agent_%d" % i,
                name=env_name + "agent_%d" % i, agents_number=env.n,
                common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape, act_space_n=env.action_space,
                model=model, lstm_model=lstm, cnn_model=cnn, lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                reuse=reuse, session=session,  agent_index=i, local_q_func=(arglist.adv_policy == 'ddpg'), args=arglist))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                actor_scope=actor_scope + "agent_%d" % i,
                name=env_name + "agent_%d" % i, agents_number=env.n,
                common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape, act_space_n=env.action_space,
                model=model, lstm_model=lstm, cnn_model=cnn, lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                reuse=reuse, session=session,  agent_index=i, local_q_func=(arglist.adv_policy == 'ddpg'), args=arglist))
    return trainers


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        l1 = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        l2 = layers.fully_connected(l1, num_outputs=num_units, activation_fn=tf.nn.relu)
        l3 = layers.fully_connected(l2, num_outputs=num_outputs, activation_fn=None)
        outupts = l3
        return outupts


def cnn_model(inputs, reuse=tf.AUTO_REUSE, scope='cnn', padding='SAME'):
    with tf.variable_scope(scope, reuse=reuse):
        l1 = tf.layers.conv2d(inputs, 16, 3, activation='relu', strides=2, padding=padding)
        l2 = tf.layers.conv2d(l1, 32, 3, activation='relu', strides=2, padding=padding)
        l3 = tf.layers.conv2d(l2, 64, 3, activation='relu', strides=2, padding=padding)
        bn = tf.layers.batch_normalization(l3)
        
        temp = 64 * bn.shape[1] * bn.shape[2]
        outputs = tf.reshape(bn, [-1, temp])
    return outputs


#  lstm模型
def lstm_model(common_obs, inputs, reuse=tf.AUTO_REUSE, num_units=(64, 32), scope="l"):
    # inputs.shape: [batch_size, time_steps, agents_number, shape]
    observation_n = []
    for i in range(len(inputs)):
        x = tf.transpose(inputs[i], (1, 0, 2))
        x = tf.concat((common_obs, x), 2)
        with tf.variable_scope(scope, reuse=reuse):
            cells = [rnn.LSTMCell(lstm_size, forget_bias=1, state_is_tuple=True) for lstm_size in num_units]
            cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
            with tf.variable_scope("Multi_Layer_RNN"):
                cell_outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            outputs = cell_outputs[-1:, :, :]
            outputs = tf.squeeze(outputs, 0)
            observation_n.append(outputs)
    return observation_n


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
