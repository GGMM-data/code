import gym
import time

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    total_reward = 0.0
    total_steps = 0
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

