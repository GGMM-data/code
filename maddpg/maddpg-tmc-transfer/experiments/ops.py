import tensorflow as tf
import os
import sys
cwd = os.getcwd()
path = cwd + "/../"
sys.path.append(path)

from model.trainer.maddpg_actor_trainer import MADDPGAgentTrainer as ACTOR_TRAINER
from model.trainer.maddpg_critic_trainer import MADDPGAgentTrainer as CRITIC_TRAINER
from model.trainer.maddpg_actor_act import MADDPGAgentTrainer as ACT
import tensorflow.nn.rnn_cell as rnn
import tensorflow.contrib.layers as layers
import h5py
import numpy as np
import time


def get_trainers(env, env_name, num_adversaries, obs_shape_n, arglist, actors=None, actor_env_name=None, type=0):
    trainer = None
    trainers = []
    lstm = lstm_model
    model = mlp_model

    if type == 0:
        trainer = ACT
        for i in range(num_adversaries):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.adv_policy == 'ddpg')))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.good_policy == 'ddpg')))
    elif type == 1:
        trainer = CRITIC_TRAINER
        for i in range(num_adversaries):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, actors, arglist,
                local_q_func=(arglist.adv_policy == 'ddpg')))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, actors, arglist,
                local_q_func=(arglist.good_policy == 'ddpg')))
    elif type == 2:
        trainer = ACTOR_TRAINER
        for i in range(num_adversaries):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i,
                actor_env_name + "agent_%d" % i, arglist,
                local_q_func=(arglist.adv_policy == 'ddpg')))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                env_name + "agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i,
                actor_env_name + "agent_%d" % i, arglist,
                local_q_func=(arglist.good_policy == 'ddpg')))
    
    return trainers


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def sample_map(path):
    f = h5py.File(path, "r")
    data = f['data'][:]
    f.close()
    data_shape = data.shape
    index = np.random.randint(0, data_shape[0])
    map = np.sum(data[index], 2)
    return map


def dimension_reduction(inputs, num_units=256, scope="dimension_reduction", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = layers.fully_connected(inputs, num_outputs=num_units * 4, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        return out


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


def lstm_model2(inputs, reuse=False, layers_number=2, num_units=256, scope="l"):
    shape = inputs[0].shape
    observation_n = []
    for i in range(len(inputs)):
        obs = inputs[i]
        if not reuse and i == 0:
            reuse = False
        else:
            reuse = True
        x = []
        with tf.variable_scope(scope, reuse=reuse):
            for j in range(shape[2]):
                dr_reuse = True
                if j == 0 and not reuse:
                    dr_reuse = False
                out = layers.fully_connected(obs[:, :, j], num_outputs=num_units * 4, activation_fn=tf.nn.relu,
                                             scope="first", reuse=dr_reuse)
                out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu,
                                             scope="second", reuse=dr_reuse)
                x.append(tf.expand_dims(out, 2))
            x = tf.concat(x, 2)
            lstm_size = x.shape[1]

        # dimension reduction 3096->1024->256
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.transpose(x, (2, 0, 1))  # (time_steps, batch_size, state_size)
            lstm_cell = rnn.BasicLSTMCell(lstm_size, forget_bias=1, state_is_tuple=True)
            cell = rnn.MultiRNNCell([lstm_cell] * layers_number, state_is_tuple=True)
            with tf.variable_scope("Multi_Layer_RNN"):
                cell_outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            outputs = cell_outputs[-1:, :, :]
            outputs = tf.squeeze(outputs, 0)
            observation_n.append(outputs)
    return observation_n


# dimension reduction + lstm模型
# inputs: list of [batch_size, dim, time_step]
def lstm_model3(inputs, reuse=False, layers_number=2, num_units=256, scope="l"):
    shape = inputs[0].shape
    observation_n = []
    for i in range(len(inputs)):
        obs = inputs[i]
        if not reuse and i == 0:
            reuse = False
        else:
            reuse = True
        x = []
        with tf.variable_scope(scope, reuse=reuse):
            for j in range(shape[2]):
                dr_reuse = True
                if j == 0 and not reuse:
                    dr_reuse = False
                out = layers.fully_connected(obs[:, :, j], num_outputs=num_units * 4, activation_fn=tf.nn.relu,
                                             scope="first", reuse=dr_reuse)
                out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu,
                                             scope="second", reuse=dr_reuse)
                x.append(tf.expand_dims(out, 2))
            x = tf.concat(x, 2)
            lstm_size = x.shape[1]

        # dimension reduction 3096->1024->256
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.transpose(x, (2, 0, 1))  # (time_steps, batch_size, state_size)
            lstm_cell = rnn.BasicLSTMCell(lstm_size, forget_bias=1, state_is_tuple=True)
            cell = rnn.MultiRNNCell([lstm_cell] * layers_number, state_is_tuple=True)
            with tf.variable_scope("Multi_Layer_RNN"):
                cell_outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            outputs = cell_outputs[-1:, :, :]
            outputs = tf.squeeze(outputs, 0)
            observation_n.append(outputs)
    return observation_n


def q_model(inputs, num_outputs, scope, reuse=False,  num_units=64):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = inputs
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, benchmark=False):
    from multiagent.environment_uav import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env





def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


def time_begin():
    return time.time()


def time_end(begin_time, info):
    print(info)
    return time.time() - begin_time