import numpy as np
import matplotlib.pyplot as plt


class N_step_TD(object):
    def __init__(self):
        self.states = np.arange(21)
        self.ACTIONS = [-1, 1]
        self.q_values = np.ones((len(self.states), len(self.ACTIONS)))
        
    def learn(self, n, alpha, q)

    def figure_7_2(self):
        runs = 100
        episodes = 10
        for r in range(runs):
            for e in range(episodes):
                self.lZZe

if __name__ == "__main__":
    
