import numpy as np
import matplotlib.pyplot as plt


class N_step_TD(object):
    def __init__(self):
        self.states = np.arange(21)
        self.ACTIONS = [-1, 1]
        
    def learn(self, n, alpha, q):
        

    def figure_7_2(self, n, alpha):
        runs = 100
        episodes = 10
        q_values = np.zeros((len(self.states), len(self.ACTIONS)))
        for r in range(runs):
            current_q = np.copy(q_values)
            for e in range(episodes):
                self.learn(r, n, alpha, current_q)
            
            

if __name__ == "__main__":
    
