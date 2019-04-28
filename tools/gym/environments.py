import gym
import numpy as np
import time

env = gym.make("PongDeterministic-v4")
state = env.reset()
print(type(state))
while True:
    state, reward, done, _ = env.step(np.random.randint(env.action_space.n))
    env.render()
    if done:
        print("Done")
        time.sleep(3)
        env.reset()

