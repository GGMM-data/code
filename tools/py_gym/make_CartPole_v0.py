import gym

env = gym.make("CartPole-v0")
print(type(env.action_space))
print(env.action_space.n)
print(env.action_space.shape)
print(type(env.observation_space))
print(env.observation_space.shape)

s_0 = env.reset()
s_1, r_1, done_1, info_1 = env.step(0)
print(s_0)
print(s_1, r_1)
