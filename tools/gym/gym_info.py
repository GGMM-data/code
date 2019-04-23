import gym
import numpy as np

from gym import envs
envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)

# env = gym.make('CartPole-v0')
env = gym.make('Breakout-v4')
obs = env.reset()
done = False
steps = 0

while not done:
    obs, rew, done, info = env.step(np.random.randint(env.action_space.n))
    steps += 1
    env.render()
    print(done)
    print(info)
    if done:
        print("done == True")
        print("info: {}".format(info))
print("steps: {}".format(steps))