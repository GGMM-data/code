import gym
import random
import time


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, environment, epsilon=-1.1):
        super(RandomActionWrapper, self).__init__(environment)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            print("Random")
            return self.env.action_space.sample()
            # return self.environment.action_space.sample()
        else:
            # print("Not random")
            pass
        return action

    def reverse_action(self, action):
        pass


if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    env = RandomActionWrapper(environment)
    
    obs = env.reset()
    episode = 0
    total_steps = 0
    total_reward = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
        total_reward += reward
        total_steps += 1
        env.render()
        time.sleep(0.1)
        if done:
            print("Episode %d done in %d steps, total reward %.1f" %(episode, total_steps, total_reward))
            time.sleep(0.1)
            env.reset()
            episode += 1
            total_reward = 0

