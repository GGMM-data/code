import numpy as np
import matplotlib.pyplot as plt

## 这个例子中得到的经验是，多运行几次得到的结果比一次结果运行很久要好的多。

class Base(object):
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=1):
        # 1.hyper parameter
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # 2.1 actions
        self.ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # 2.2 state and action dim
        self.x_dim = 4
        self.y_dim = 12
        self.action_dim = len(self.ACTIONS)

        # 2.3 special states 
        self.start_state = (3, 0)
        self.goal_state = (3, 11)

        # 2.4 penalty states
        self.penalty_states = []
        for i in range(1, 11):
            self.penalty_states.append((3, i))

        # 2.5 q value
        self.action_values = np.zeros((self.x_dim, self.y_dim, self.action_dim))
        # 3. rewards list
        self.name = "base"

    def step(self, action):
        reward = -1
        state = (self.state[0] + action[0], self.state[1] + action[1])
        if state in self.penalty_states:
            reward = -100
            state = self.start_state
        state = self.check_boundary(state)
        return state, reward

    def epsilon_greedy_policy(self, state):
        if np.random.binomial(1, self.epsilon):
            action = np.random.choice(self.action_dim)
        else:
            q = self.action_values[state[0], state[1], :]
            action = np.random.choice([a for a, v in enumerate(q) if v == np.max(q)])
        return action

    def greedy_policy(self, state):
        q = self.action_values[state[0],state[1], :]
        action = np.random.choice([a for a,v in enumerate(q) if v == np.max(q)])
        return action

    def check_boundary(self, state):
        x, y = state[0], state[1]
        if state[0] < 0:
            x = 0
        if state[0] > self.x_dim - 1:
            x = self.x_dim - 1
        if state[1] < 0:
            y = 0
        if state[1] > self.y_dim - 1:
            y = self.y_dim -1
        return (x,y)
        
    def show(self, runs, rewards):
        plt.figure()
        plt.plot(rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        plt.ylim([-100, 0])
        plt.savefig("./images/example_6_6_" + self.name + "_runs_"+str(runs)+ ".png")
        plt.close()


class Sarsa(Base):
    def __init__(self, alpha=0.5, epsilon=0.1):
        super(Sarsa, self).__init__(alpha, epsilon)
        self.name = "sarsa"

    def sarsa(self, runs, episodes, expected=False):
        if expected:
            self.name = "expected_sarsa"
        else:
            self.name = "sarsa"
            
        total_rewards = np.ones(episodes)
        for r in range(runs):
            self.action_values = np.zeros((self.x_dim, self.y_dim, self.action_dim))
            sum_rewards_list = []
            for i in range(episodes):
                self.state = self.start_state
                sum_rewards = 0
    
                action_index = self.epsilon_greedy_policy(self.state)
                while True:
                    next_state, reward = self.step(self.ACTIONS[action_index])
                    sum_rewards += reward
    
                    # update state action value
                    next_action_index = self.epsilon_greedy_policy(next_state)
                    x, y = self.state
                    nx, ny = next_state
                    if expected:
                        target = 0.0
                        q_values = self.action_values[nx, ny, :]
                        best_actions = np.argwhere(q_values == np.max(q_values))
                        for a in range(self.action_dim):
                            if a in best_actions:
                                target += ((1 - self.epsilon)/len(best_actions) + self.epsilon/self.action_dim) * q_values[a]
                            else:
                                target += self.epsilon/self.action_dim * q_values[a]
                    else:
                        target = self.gamma * self.action_values[nx][ny][next_action_index]
                    self.action_values[x][y][action_index] += self.alpha*(reward + target - self.action_values[x][y][action_index])
    
                    # move to next time step
                    self.state = next_state
                    action_index = next_action_index

                    if self.state[0] == self.goal_state[0] and self.state[1] == self.goal_state[1]:
                        sum_rewards_list.append(sum_rewards)
                        break

            total_rewards += np.array(sum_rewards_list)
        print("%s alpha %.3f, run %d times Done." % (self.name, self.alpha, runs))
        total_rewards /= runs
        return total_rewards
 

class Q_learning(Base):
    def __init__(self, alpha=0.5, epsilon=0.1):
        super(Q_learning, self).__init__(alpha, epsilon)
        self.name = "Q-learning"

    def q_learning(self, runs, episodes):
        total_rewards = np.ones(episodes)
        for r in range(runs):
            self.action_values = np.zeros((self.x_dim, self.y_dim, self.action_dim))
            sum_rewards_list = []
            for i in range(episodes):
                self.state = self.start_state
                sum_rewards = 0
    
                while True:
                    # action selection
                    action_index = self.epsilon_greedy_policy(self.state)
                    next_state, reward = self.step(self.ACTIONS[action_index])
                    sum_rewards += reward
    
                    # update state action value
                    x, y = self.state
                    nx, ny = next_state
                    q = self.action_values[nx, ny, :]
                    self.action_values[x][y][action_index] += self.alpha * (reward + self.gamma * np.max(q) -
                                                                            self.action_values[x][y][action_index] )
    
                    # move to next time step
                    self.state = next_state
                    if self.state[0] == self.goal_state[0] and self.state[1] == self.goal_state[1]:
                        sum_rewards_list.append(sum_rewards)
                        break
    
            total_rewards += np.array(sum_rewards_list)
        total_rewards /= runs
        print("%s alpha %.3f, run %d times Done." % (self.name, self.alpha, runs))
        return total_rewards
        
   
def example_6_6():
    runs = 10
    begin_run = runs
    episodes = 500

    q = Q_learning()
    for r in range(begin_run, runs+1):
        total_rewards = q.q_learning(r, episodes)
        q.show(r, total_rewards)

    sarsa = Sarsa()
    for r in range(begin_run, runs+1):
        total_rewards = sarsa.sarsa(r, episodes, expected=True)
        sarsa.show(r, total_rewards)


def figure_6_3():
    runs = 10
    interim_episodes = 100
    asymptotic_episodes = 10000
    alpha_list = np.arange(0.1, 1.1, 0.1)

    asymptotic_q_rewards = []
    interim_q_rewards = []
    asymptotic_sarsa_rewards = []
    interim_sarsa_rewards = []
    asymptotic_expected_sarsa_rewards = []
    interim_expected_sarsa_rewards = []

    for alpha in alpha_list:
        q = Q_learning(alpha)
        interim_episode_rewards = q.q_learning(runs, interim_episodes)
        interim_q_rewards.append(np.mean(interim_episode_rewards))

    for alpha in np.arange(0.1, 1.1, 0.1):
        sarsa = Sarsa(alpha)
        interim_episode_rewards = sarsa.sarsa(runs, interim_episodes, expected=False)
        interim_sarsa_rewards.append(np.mean(interim_episode_rewards))

    for alpha in np.arange(0.1, 1.1, 0.1):
        expected_sarsa = Sarsa(alpha)
        interim_episode_rewards = expected_sarsa.sarsa(runs, interim_episodes, expected=True)
        interim_expected_sarsa_rewards.append(np.mean(interim_episode_rewards))

    plt.plot(alpha_list, interim_q_rewards, label='Q-learning')
    plt.plot(alpha_list, interim_sarsa_rewards, label='Sarsa')
    plt.plot(alpha_list, interim_expected_sarsa_rewards, label='Expected Sarsa')
    plt.legend()
    plt.savefig("./images/figure_6_3.png")
    plt.show()
    

if __name__ == "__main__":
    # example_6_6()
    figure_6_3()

