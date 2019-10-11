import gym
import time
import ptan


class ToyEnv(gym.Env):
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=3)
        self.step_index = 0
        
    def reset(self):
        self.step_index = 0
        return self.step_index
    
    def step(self, action):
        is_done = self.step_index >= 10
        print(self.step_index)
        if is_done:
            return self.step_index % self.observation_space.n, 0.0, is_done, {}
        self.step_index += action
        return self.step_index % self.observation_space.n, float(action), self.step_index == 10, {}


class DullAgent(ptan.agent.BaseAgent):
    def __init__(self, action: int):
        self.action = action

    def __call__(self, observations, state=None):
        return [self.action for _ in observations], state

if __name__ == "__main__":
    # env
    env = ToyEnv()

    # agent
    agent = DullAgent(action=1)

    # 
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=5)
    for exp in exp_source:
        print(exp)
        if exp[0].done:
            break
    
#     init_state = env.reset()
#     while True:
#         state, reward, done, info = env.step(env.action_space.sample())
#         #state, action, reward, info = env.step(0)
#         print(state, reward, done, info)
#         time.sleep(1)
#         if done:
#             env.reset()
#     
