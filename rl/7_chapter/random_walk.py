import numpy as np
import matplotlib.pyplot as plt


class N_step_TD(object):
    def __init__(self):
        self.states = np.arange(21)
        self.ACTIONS = [-1, 1]
        self.start_state = len(self.states)/2
        self.terminal_state = [0, 20]

    def policy(self, state):
        action = np.random.binomial(1, )
        return action
        
        
    def learn(self, n, alpha, q)
        state = self.start_state
        while True:
            action = self.policy(state)
            next_state, reward = self.step(action)
            
        

    def figure_7_2(self, n, alpha):
        runs = 100
        episodes = 10
        for r in range(runs):
            self.q_values = np.ones((len(self.states), len(self.ACTIONS)))
            for e in range(episodes):
                self.learn(n, alpha, q)

if __name__ == "__main__":
    
