"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import random
import numpy as np


class ReplayBuffer:
  def __init__(self, config, obs_shape, act_space):
    self.cnn_format = config.cnn_format
    self.buffer_size = config.buffer_size
    self.actions = np.empty((self.buffer_size, act_space,), dtype=np.float16)
    self.rewards = np.empty(self.buffer_size, dtype=np.float16)
    self.observations = np.empty((self.buffer_size,) + obs_shape, dtype=np.float16)
    self.terminals = np.empty(self.buffer_size, dtype=np.bool)
    self.dims = obs_shape
    self.history_length = config.history_length
    self.batch_size = config.batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.states = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
    self.next_states = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)

  def __len__(self):
    return self.count

  def add(self, obs, action, reward, done, terminal):
    assert obs.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.observations[self.current, ...] = obs
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.buffer_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      return self.observations[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.observations[indexes, ...]

  def sample(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.states[len(indexes), ...] = self.getState(index - 1)
      self.next_states[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.states, (0, 2, 1)), actions, \
        rewards, np.transpose(self.next_states, (0, 2, 1)), terminals
    else:
      return self.states, actions, rewards, self.next_states, terminals
