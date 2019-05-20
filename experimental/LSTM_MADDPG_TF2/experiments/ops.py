import tensorflow as tf
from LSTM_MADDPG_TF2.model.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.nn.rnn_cell as rnn
import tensorflow.contrib.layers as layers


# lstm模型
# inputs: list of [batch_size, dim, time_step]
def lstm_model(inputs, history_length, batch_size, reuse=False, layers_number=2, scope="l", rnn_cell=None):
    shape = inputs[0].shape
    lstm_size = shape[1]
    observation_n = []
    for i in range(len(inputs)):
        obs = inputs[i]
        if not reuse:
            if i == 0:
                reuse = False
            else:
                reuse = True
        with tf.variable_scope(scope, reuse=reuse):
            x = obs
            x = tf.transpose(x, (2, 0, 1))  # (time_steps, batch_size,state_size)
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


# multi perception layers
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from LSTM_MADDPG_TF2.multiagent.environment_uav import MultiAgentEnv
    import LSTM_MADDPG_TF2.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, env_name, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    lstm = lstm_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            env_name+"agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            env_name+"agent_%d" % i, model, lstm, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


def conv2d(inputs, output_dim, kernel_size, stride, initializer, activation_fn,
           padding='VALID', data_format='NHWC', name="conv2d", reuse=False):
    kernel_shape = None
    with tf.variable_scope(name, reuse=reuse):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], inputs.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], inputs.get_shape()[-1], output_dim ]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(inputs, w, stride, padding, data_format=data_format)

        b = tf.get_variable('b', [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format=data_format)

    if activation_fn is not None:
        out = activation_fn(out)
    return out, w, b


def linear(inputs, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear', reuse=False):
    shape = inputs.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], tf.float32, tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(inputs, w), b)

    if activation_fn != None:
        out = activation_fn(out)
    return out, w, b

# tf.train.RMSPropOptimizer()







