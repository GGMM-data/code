import numpy as np
import matplotlib.pyplot as plt

# 这里出现的问题是计算target的时候，多加一个action 维度，然而这个action不可能被采用，它的q值也不会被更新，一直都是0，
# 进行max的时候，就没办法取负数，最后得到的全是正数，而事实上所有action的q值都是负的。

class Double_Q_Learning(object):
    def __init__(self, epsilon=0.1, alpha=0.1):
        self.epsilon = epsilon
        self.alpha = alpha
        self.states = [0, 1, 2, 3]
        self.start_state = 2
        self.terminal_states = [0, 3]
        self.ACTIONS = [-1, 1]
        self.q1_values = [np.ones(1), np.ones(1), np.ones(2), np.ones(1)]
        self.q2_values = [np.ones(1), np.ones(1), np.ones(2), np.ones(1)]

    def learn(self, runs, episodes, double_learning=False):
        total_left_action = np.zeros(episodes)
        for r in range(runs):
            self.q1_values = [np.ones(1), np.ones(1), np.ones(2), np.ones(1)]
            self.q2_values = [np.ones(1), np.ones(1), np.ones(2), np.ones(1)]

            left_action = np.zeros(episodes)
            for i in range(episodes):
                state = self.start_state
                while True:
                    action = self.epsilon_greedy_policy(state, double_learning)
                    if state == 2:
                        left_action[i] = left_action[i-1] + (1 - action)
                    next_state, reward = self.step(state, action)
        
                    if double_learning:
                        if np.random.binomial(1, 0.5):
                            active_q = self.q1_values
                            target_q = self.q2_values

                        else:
                            active_q = self.q2_values
                            target_q = self.q1_values
    
                        q_value = active_q[next_state]
                        max_action = np.random.choice([a for a, v in enumerate(q_value) if v == np.max(q_value)])
                        target = target_q[next_state][max_action]
                    else:
                        active_q = self.q1_values
                        target = np.max(active_q[next_state])
                        if target == 0:
                            print("hello")
                    active_q[state][action] += self.alpha * (
                                    reward + target - active_q[state][action])
                    
                    state = next_state
                    if state in self.terminal_states:
                        break
                        
            total_left_action += left_action
        total_left_action /= runs
        
        return total_left_action
            
    def epsilon_greedy_policy(self, state, double_learning=False):
        if state == 1:
            action = 0
        elif state == 2 and np.random.binomial(1, self.epsilon) == 0:
            if double_learning:
                q_value = (self.q1_values[2] + self.q2_values[2])/2
            else:
                q_value = self.q1_values[2]
            action = np.random.choice([a for a, v in enumerate(q_value) if v == np.max(q_value)])
        else:
            action = np.random.choice(len(self.ACTIONS))
        return action
        
    def step(self, state, action):
        if state == 1:
            reward = np.random.normal(-0.1, 1)
        else:
            reward = 0
            
        next_state = state + self.ACTIONS[action]
        return next_state, reward


if __name__ == "__main__":
    dq = Double_Q_Learning()
    runs = 1000
    episodes = 300
    q_total_left_action = dq.learn(runs=runs, episodes=episodes, double_learning=False)
    dq_total_left_action = dq.learn(runs=runs, episodes=episodes, double_learning=True)
    plt.plot(dq_total_left_action / (range(1, episodes + 1)), label="Double Q-learning")
    plt.plot(q_total_left_action / (range(1, episodes + 1)), label="Q-learning")
    plt.legend()
    plt.savefig("./images/figure_6_5.png")
    plt.show()
