import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class N_step_TD(object):
    def __init__(self):
        self.states = np.arange(21)
        self.ACTIONS = [-1, 1]
        self.start_state = 10
        self.terminal_states = [0, 20]
        self.true_values = np.arange(-20, 22, 2) / 20.0
        self.true_values[0] = self.true_values[-1] = 0
        self.gamma = 1.0


    def policy(self, state):
        action = np.random.binomial(1, 0.5)
        return action

    def step(self, state, action):
        next_state = state + self.ACTIONS[action]
        reward = 0
        if next_state == 0:
            reward = -1
        if next_state == 20:
            reward = 1

        return next_state, reward

    def learn(self, n, alpha, q):
        time = 0
        current_state = self.start_state
        state_list = [current_state]
        reward_list = [0]

        T = float('inf')
        while True:
            time += 1
            if time < T:
                action = self.policy(current_state)
                next_state, reward = self.step(current_state, action)

                state_list.append(next_state)
                reward_list.append(reward)
                if next_state in self.terminal_states:
                    T = time

            update_time = time - n
            # update
            if update_time >= 0:
                rewards = 0
                target = 0
                for i in range(update_time + 1, min(update_time + n, T)+1):
                    rewards += np.power(self.gamma, i - update_time - 1) * reward_list[i]
                if update_time + n <= T:
                    target = rewards + np.power(self.gamma, n) * q[state_list[update_time + n]]
                update_state = state_list[update_time]
                if update_state not in self.terminal_states:
                    q[update_state] = q[update_state] + alpha * (target - q[update_state])
            if update_time == T - 1:
                break
            current_state = next_state

    def mean_q(self, n, alpha):
        runs = 100
        episodes = 10

        q_values = np.zeros((runs, episodes))
        for r in tqdm(range(runs)):
            current_q = np.zeros((len(self.states)))
            for e in range(episodes):
                self.learn(n, alpha, current_q)
                q_values[r][e] = np.sqrt(np.sum(np.square(current_q - self.true_values))/len(self.true_values))

        return np.mean(q_values)

    def figure_7_2(self):
        n_len = 10
        alpha_len = 10
        results = np.zeros((n_len, alpha_len))

        for i, n in enumerate([np.power(2, i) for i in range(n_len)]):
            for j, alpha in enumerate(np.arange(0.1, 1.1, 0.1)):
                results[i][j] = self.mean_q(n, alpha)
        plt.figure()
        for i in range(n_len):
            plt.plot(results[i], label=str(np.power(2, i)))
        plt.legend()
        plt.savefig("figure_7_2.png")
        plt.show()


if __name__ == "__main__":
    n_step_td = N_step_TD()
    n_step_td.figure_7_2()

