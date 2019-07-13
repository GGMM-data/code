import numpy as np
import matplotlib.pyplot as plt

# states = ['A', 'B', 'C', 'D', 'E']
states = [0, 1, 2, 3, 4, 5, 6]
probabilities = [0.5, 0.5]
actions = [-1, 1]
true_values = [0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6]


def mc(v, alpha=0.1):
    terminal = False
    state = 3
    trajectory = [state]
    returns = [0]
    while not terminal:
        action = np.random.binomial(1, probabilities[0])
        next_state = state + actions[action]
        if next_state == states[0] or next_state == states[-1]:
            if next_state == states[0]:
                returns = 0.0
            if next_state == states[-1]:
                returns = 1.0
            terminal = True
        trajectory.append(state)
    for state in trajectory:
        v[state] = v[state] + alpha * (returns - v[state]) #在这个问题中，所有state的retrun都是一样的
    return trajectory, [returns]*(len(trajectory) - 1)
    

# 这里的经验，把reward全部设为0，然后state value，右边设置为1
def td(v, alpha=0.1):
    terminal = False
    state = 3
    trajectory = [state]
    rewards = [0]
    while not terminal:
        action = np.random.binomial(1, probabilities[0])
        next_state = state + actions[action]
        reward = 0
        v[state] = v[state] + alpha * (reward + v[next_state] - v[state])
        state = next_state
        if next_state == states[0] or next_state == states[-1]:
            terminal = True
        trajectory.append(state)
        rewards.append(reward)
    return trajectory, rewards


def compute_state_value():
    pass


def rmse():
    pass


def example_6_2():
    numbers = [0, 1, 10, 100] 
    for i in numbers:
        v = td(i)
        plt.plot(range(len(states)), v)
        
        print(v)
    plt.plot(range(len(states)), true_values)
    plt.show()


if __name__ == "__main__":
    example_6_2()
