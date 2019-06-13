import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_uav', 7, 'the number of UAVS')
flags.DEFINE_integer('size_map', 10, 'the size of map')
flags.DEFINE_float('radius', 3., 'sensing range')
flags.DEFINE_float('constrain', 5., 'connectivity constraint between 2 uavs')
flags.DEFINE_integer('max_epoch', 500, 'the max epoch')
flags.DEFINE_float('max_speed', 1., 'max value of speed')
flags.DEFINE_float('factor', 1.0, 'energy(honor) / energy(max distance)')

flags.DEFINE_float('map_scale_rate', 5., 'gym map transform my map')
flags.DEFINE_integer('penalty', 10, 'the penalty when uavs are out of map')
flags.DEFINE_integer('penalty_disconnected', 1, 'the penalty when uavs are out of map')
flags.DEFINE_float('map_constrain', 4.5, 'over map constrain')
# sensitivity = np.sqrt(2) / 2
sensitivity = 5.
flags.DEFINE_float('action_sensitivity', sensitivity, 'action sensitivity')
flags.DEFINE_boolean('greedy_action', False, 'uav act based greedy algorithm')
flags.DEFINE_boolean('random_action', False, 'uav choose random action')
flags.DEFINE_float('greedy_act_dis', 0.1, 'random action move distance')
