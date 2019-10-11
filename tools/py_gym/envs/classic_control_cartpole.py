import gym
import time
# pip install gym[classic_control]

env = gym.make('CartPole-v0')
env.reset()
episode = 1
while True:
    obs, reward, done, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        print("============Episode {} done.==================".format(episode))
        if episode >= 2:
            break
        episode += 1
        time.sleep(2)
        env.reset()

