import numpy as np
import tensorflow as tf
from collections import deque
import random
import gym

LEARNING_RATE =  0.0001 
GAMMA = 0.9 
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
EPSILON = 0.01
HIDDEN_UNITS = 20

class DQN:
  def __init__(self, env):
    self.replay_buffer = deque()
    self.state_dim = env.observation_space.shape[0] 
    self.action_dim  = env.action_space.n
    self.hidden_dim = HIDDEN_UNITS
    
    self.lr = LEARNING_RATE
    self.gamma = GAMMA
    self.batch_size = BATCH_SIZE
    self.replay_buffer_size = REPLAY_BUFFER_SIZE
    self.epsilon = EPSILON

    # fully-connected layer
    w1 = tf.Variable(tf.truncated_normal(shape=[self.state_dim, self.hidden_dim]))
    b1 = tf.Variable(tf.constant(0.01, shape=[self.hidden_dim]))
    w2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_dim, self.action_dim]))
    b2 = tf.Variable(tf.constant(0.01, shape=[self.action_dim]))
   
    # state input
    self.state_input = tf.placeholder("float", [None, self.state_dim])
    print(self.state_input.shape)
    # hidden layer
    hidden_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
    # predicted q value
    self.q = tf.matmul(hidden_layer, w2) + b2
    
    # action input
    self.action_input = tf.placeholder("float", [None, self.action_dim])
    # predicted q value 
    # action is one hot representation
    self.predicted_q = tf.reduce_sum(tf.multiply(self.q, self.action_input), reduction_indices=1) 
    # true value
    self.y = tf.placeholder("float", [None]) 
    # loss function
    self.loss = tf.reduce_mean(tf.square(self.y - self.predicted_q)) 

    # learning rate
    self.lr = LEARNING_RATE
    # optimizer
    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())

  def greedy_action(self, state):
    if random.random() < self.epsilon:
      return np.random.randint(0, self.action_dim - 1)
    else:
      q = self.q.eval(feed_dict={self.state_input: [state]})
      return np.argmax(q)

  def action(self, state):
    q = self.q.eval(feed_dict={self.state_input: [state]})
    return np.argmax(q)

  def experience(self, state, action, reward, next_state, done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
    if len(self.replay_buffer) > self.replay_buffer_size:
       self.replay_buffer.popleft()

  def update(self):
    if len(self.replay_buffer) < self.batch_size:
       return

    batch = random.sample(self.replay_buffer, self.batch_size) 
    state = [data[0] for data in batch]
    action = [data[1] for data in batch]
    reward = [data[2] for data in batch]
    next_state = [data[3] for data in batch]

    next_state_values = self.q.eval(feed_dict={self.state_input: next_state})
    y = []

    # calculate true q value
    for i in range(self.batch_size):
      if batch[i][4]:
        y.append(reward[i])
      else:
        next_state_value = np.max(next_state_values[i])
        y.append(reward[i] + self.gamma * next_state_value)
    self.optimizer.run(feed_dict={self.state_input: state, self.action_input:action, self.y: y}) 


# ENV_NAME = "CartPole-v0"
#
# episodes = 3000
# test_episodes = 100
# steps = 300
# test_steps = 300
#
# def train():
#   # initialize environment
#   env = gym.make(ENV_NAME)
#   # create a agent
#   agent = DQN(env)
#
#   # begin to train
#   for i in range(episodes):
#     # reset environment
#     state = env.reset()
#     for step in range(steps):
#       # choose action
#       action = agent.greedy_action(state)
#       # run a step
#       next_state, reward, done, _ = env.step(action)
#       # reset reward
#       reward = -1 if done else 0.1
#       # add transition to replay buffer
#       agent.experience(state, action, reward, next_state, done)
#
#       # update parameters
#       agent.update()
#
#       # move to next state
#       state = next_state
#       if done:
#         break
#
#     if i % 100 == 0:
#       test(agent, env, i)
#
# def test(agent, env, episode):
#   total_reward = 0
#   for i in range(test_episodes):
#     state = env.reset()
#     for step in range(test_steps):
#       # env.render()
#       action = agent.action(state)
#       next_state, reward, done, _ = env.step(action)
#       total_reward += reward
#       if done:
#         break
#   average_reward = total_reward / test_episodes
#   print("episode: ", episode, "average reward", average_reward)
#
#
# if __name__ == "__main__":
#   train()
