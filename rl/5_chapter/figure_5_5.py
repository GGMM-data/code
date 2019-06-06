import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

ACTION_BACK = 0
ACTION_END = 1

class Env:
    def __init__(self):
        self.prob = 0.9

    def step(self, action):
        """
        0: left, 1: right
        """
        next_state = 0
        reward = 0.0
        if action == ACTION_BACK:
            result = np.random.binomial(1, 0.9)     #
            if result == 0:     #
                next_state = -1
                reward = 1.0
        elif action == ACTION_END:
            next_state = -1
        return next_state, reward

        
class policy:
    def __init__(self, left_prob):
        self.left_prob = left_prob

    def action(self):
        action = np.random.binomial(1, 1 - self.left_prob)
        return action


def figure_5_4():
    env = Env()
    target_action_prob = 1.0
    behaviour_action_prob = 0.5
    target_policy = policy(target_action_prob)
    behaviour_policy = policy(behaviour_action_prob)
    ratios_list = []
    returns_list = []
    values_list = []
    runs = 10
    episodes = 100000
    for i in range(runs):
        for j in range(episodes):
            returns = 0
            ratio = 1.0
            # run every episode
            while True:
                action = behaviour_policy.action()
                next_state, reward = env.step(action)
                returns += reward

                if action == ACTION_BACK:
                    ratio = ratio * target_policy.left_prob / behaviour_policy.left_prob
                elif action == ACTION_END:   #
                    target_action_end_prob = 1 - target_policy.left_prob
                    behaviour_action_end_prob = 1 - behaviour_policy.left_prob
                    ratio = ratio * target_action_end_prob / behaviour_action_end_prob
                    break

                if next_state == -1:
                    break
                
            ratios_list.append(ratio)
            returns_list.append(returns)
            values_list.append(ratio * returns)
        values = np.add.accumulate(values_list)
        estimations = np.asarray(values) / np.arange(1, episodes + 1)
        plt.plot(estimations)
        
        ratios_list = []
        returns_list = []
        values_list = []
    
    # plt.ion()
    # plt.pause(0.1)
    # plt.ioff()
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.savefig('../images/figure_5_4.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    figure_5_4()
