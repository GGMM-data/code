import gym
import numpy as np
import time

# 1.创建一个env
env = gym.make("Breakout-v4")

print(env.action_space.shape)
# 2.reset环境
state = env.reset()
print(type(state))

# 3.step by step
while True:
    #state, reward, done, _ = env.step(np.random.randint(env.action_space.n))
    state, reward, done, info = env.step(env.action_space.sample())
    time.sleep(0.1)
    # 这里的Info是还有几条命，每次浪费一条命就减去$10$
    # print(info)
    if reward != 0.0:
        print(reward)
    # render出来的iamge，左边是得分，右边是还有几条命。
    env.render()
    if done:
        print("Done")
        time.sleep(0.3)
        env.reset()

