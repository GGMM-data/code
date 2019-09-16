import gym
import numpy as np
import time

# 1.创建一个env
env = gym.make("Breakout-v4")

# 2.reset环境
state = env.reset()
print(type(state))

# 3.step by step
while True:
    #state, reward, done, _ = env.step(np.random.randint(env.action_space.n))
    state, reward, done, _ = env.step(env.action_space.sample())
    if reward != 0.0:
        print(reward)
    env.render()
    if done:
        print("Done")
        time.sleep(0.3)
        env.reset()

