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


def sample_map(path):
    f = h5py.File(path, "r")
    data = f['data'][:]
    f.close()
    data_shape = data.shape
    index = np.random.randint(0, data_shape[0])
    map = np.sum(data[index], 2)
    return map


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, benchmark=False):
    from multiagent.environment_uav import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, env_name, num_adversaries, obs_shape_n, arglist, actors=None, actor_env_name=None,
                 lstm_scope=None, type=0, session=None):
    trainers = []
    model = mlp_model
    lstm = lstm_model
    if type == 0:
        trainer = ACT
        for i in range(num_adversaries):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.adv_policy == 'ddpg'), session=session))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.good_policy == 'ddpg'), session=session))
    elif type == 1:
        trainer = CRITIC_TRAINER
        for i in range(num_adversaries):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, actors, arglist,
                local_q_func=(arglist.adv_policy == 'ddpg'), lstm_scope=lstm_scope+ "agent_%d" % i, session=session))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, actors, arglist,
                local_q_func=(arglist.good_policy == 'ddpg'), lstm_scope=lstm_scope+ "agent_%d" % i, session=session))
    elif type == 2:
        trainer = ACTOR_TRAINER
        for i in range(num_adversaries):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i,
                actor_env_name + "agent_%d" % i, arglist,
                local_q_func=(arglist.adv_policy == 'ddpg'), session=session))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i,
                actor_env_name + "agent_%d" % i, arglist,
                local_q_func=(arglist.good_policy == 'ddpg'), session=session))
    return trainers


#  lstm模型
def lstm_model(inputs, reuse=False, num_units=(64, 32), scope="l"):
    debug = False
    if debug:
        import time
        t = time.time()
    observation_n = []
    for i in range(len(inputs)):
        x = inputs[i]
        if not reuse and i == 0:
            reuse = False
        else:
            reuse = True

        with tf.variable_scope(scope, reuse=reuse):
            x = tf.transpose(x, (2, 0, 1))  # (time_steps, batch_size, state_size)
            cells = [rnn.LSTMCell(lstm_size, forget_bias=1, state_is_tuple=True) for lstm_size in num_units]
            cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
            with tf.variable_scope("Multi_Layer_RNN"):
                cell_outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            outputs = cell_outputs[-1:, :, :]
            outputs = tf.squeeze(outputs, 0)
            observation_n.append(outputs)
    if debug:
        print("lstm time: ", time.time()-t)
    return observation_n


def time_begin():
    return time.time()


def time_end(begin_time, info):
    print(info)
    return time.time() - begin_time


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
