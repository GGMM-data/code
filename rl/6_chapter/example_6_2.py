import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# states = ['A', 'B', 'C', 'D', 'E']
states = [0, 1, 2, 3, 4, 5, 6]
probabilities = [0.5, 0.5]
actions = [-1, 1]
true_values = [0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6]
values = np.ones(len(states))/2
values[0] = 0.0
values[len(states)-1] = 1.0


def mc(v, alpha=0.1):
    terminal = False
    state = 3
    trajectory = [state]
    while True:
        action = np.random.binomial(1, probabilities[0])
        next_state = state + actions[action]
        state = next_state
        trajectory.append(state)
        if next_state == states[0] or next_state == states[-1]:
            if next_state == states[0]:
                returns = 0.0
            if next_state == states[-1]:
                returns = 1.0
            break
    for state in trajectory:
        v[state] = v[state] + alpha * (returns - v[state]) #在这个问题中，所有state的retrun都是一样的
    return trajectory, [returns]*(len(trajectory) - 1)
    

# 这里的经验，把reward全部设为0，然后state value，右边设置为1
def td(v, alpha=0.1):
    state = 3
    trajectory = [state]
    rewards = [0.0]
    while True:
        action = np.random.binomial(1, probabilities[0])
        next_state = state + actions[action]
        reward = 0.0
        v[state] = v[state] + alpha * (reward + v[next_state] - v[state])
        state = next_state
        trajectory.append(state)
        rewards.append(reward)
        if next_state == states[0] or next_state == states[-1]:
            break
    return trajectory, rewards


def compute_state_value():
    numbers = [0, 1, 10, 100] 
    current_values = np.copy(values)
    plt.figure()
    for i in range(numbers[-1]+1):
        if i in numbers:
            print(i)
            plt.plot(range(len(states)), current_values, label=str(i)+" episodes")
        td(current_values, 0.1)
    print(current_values)
    plt.plot(range(len(states)), true_values, label="true values")
    plt.xlabel('state')
    plt.ylabel("estimated value")
    plt.legend()
    plt.savefig("example_6_2_td")


def rmse():
    td_alpha = [0.15, 0.1, 0.05]
    mc_alpha = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    #
    plt.figure()
    for i, alpha in enumerate(td_alpha + mc_alpha):
        total_errors = np.zeros(episodes)
        if i < len(td_alpha):
            method = "TD"
            linestyle = 'solid'
        else:
            method = "MC"
            linestyle = "dashdot"
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(values)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(true_values - current_values, 2))/5.0))
                if method == "TD":
                    td(current_values, alpha)
                else:
                    mc(current_values, alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=method+"alpha %.02f" % (alpha))
    plt.xlabel("episodes")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("example_6_2_RMSE")


if __name__ == "__main__":
    compute_state_value()
    rmse()
