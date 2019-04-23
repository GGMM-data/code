import gym
import numpy as np

env = gym.make('SpaceInvaders-v0')
env.reset()
while True:
    env.step(np.random.randint(env.action_space.n))
    env.render()