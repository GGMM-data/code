import numpy as np
import matplotlib.pyplot as plt

# 这个不对的原因是每个episode结束的条件错了，应该判断当前state是不是terminal state，如果是的话，就要break了，这个state的所有Q值都不会被更新，和初始值一样的，如果next_state是terminal_state的话，根据q值选择next_action，相当于随机选择一个更新当前state的q值。然后接下来就要break了。
# 风速搞错了，风速是平行于x轴，垂直于y轴的，跟y值有关，我用了x值。注意，这里的x轴，y轴不是数学上的，用的是数组，
# 第一维我叫做x轴，第二维叫做y轴
# 当在state s处，如果不同a的q值一样的时候，从中随机选择一个。

class Sarsa(object):
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=1.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.start_state = [3, 0]
        self.goal_state = [3, 7]

        self.windy = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.ACTIONS = [(1, 0), (-1, 0), (0, -1), (0, 1)]

        self.action_dim = len(self.ACTIONS)
        self.x_dim = 7
        self.y_dim = 10

        self.action_values = np.zeros((self.x_dim, self.y_dim, self.action_dim))

    def sarsa(self):
        episodes = 500
        steps_list = []

        for i in range(episodes):
            steps = 0
            self.state = self.start_state
            action_index = self.epsilon_greedy_policy(self.state)

            while True:
                next_state, reward = self.step(action_index)
                next_action_index = self.epsilon_greedy_policy(next_state)
                # update state action value
                self.action_values[self.state[0]][self.state[1]][action_index] += self.alpha*(reward + self.gamma * self.action_values[next_state[0]][next_state[1]][next_action_index] - self.action_values[self.state[0]][self.state[1]][action_index])

                # move to next time step
                # 不能放这里啊，放这里是啥意思，错了啊，如果放在这里，在step中已经到terminal了，还在往下一步走。
#                 if self.state == self.goal_state:
#                     break
                self.state = next_state
                action_index = next_action_index
                steps += 1
                if self.state == self.goal_state:
                    break

            steps_list.append(steps)
            print(steps)

        steps_list = np.add.accumulate(steps_list)
        self.show(steps_list)
    
    def step(self, action):
        reward = -1
        action = self.ACTIONS[action]
        state = [self.state[0] + action[0] - self.windy[self.state[1]], self.state[1] + action[1]]
        state = self.check_boundary(state)
        return state, reward

    def epsilon_greedy_policy(self, state):
        if np.random.binomial(1, self.epsilon):
            action = np.random.choice(self.action_dim)
        else:
            q_values = self.action_values[state[0], state[1], :]
            action = np.random.choice([a for a, v in enumerate(q_values) if v == np.max(q_values)])
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
            y = self.y_dim - 1
        return [x, y]
        
    def show(self, steps_list):
        plt.figure()
        plt.plot(steps_list, range(1, len(steps_list) + 1))
        plt.savefig("./images/example_6_5.png")
        plt.close()
    

if __name__ == "__main__":
    sarsa = Sarsa()
    sarsa.sarsa()

