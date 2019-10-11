import gym
import time

env = gym.make('Copy-v0')
env.reset()
episode = 1
while True:
    obs, reward, done, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        print("===============Episode {} done.=================".format(episode))
        if episode >= 2:
            break
        episode += 1
        time.sleep(0.5)
        env.reset()

