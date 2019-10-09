import gym
import random

class RandomActionWrapper(gym.ActionWrapper):
    def __int__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            print("Random")
            return self.env.action_space.sample()
        return action

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = RandomActionWrapper(env, epsilon=0.3)
    
    obs = env.reset()
    episode = 1
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        env.render()
        if done:
            print("Episode %d done in %d steps, total reward %.2f" %(episode, total_steps, total_reward))
            time.sleep(1)
            env.reset()
            episode += 1
            total_reward = 0

