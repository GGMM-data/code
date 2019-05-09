import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch
import math
import numpy as np
from model import DQN
from policy import Policy
import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
from utils import plot_rewards, plot_durations, plot_state, get_screen
from ops import select_action
import gym

def trainD(file_name="Distral_1col", list_of_envs=[GridworldEnv(4),
            GridworldEnv(5)], batch_size=128, gamma=0.999, alpha=0.9,
            beta=5, eps_start=0.9, eps_end=0.05, eps_decay=5,
            is_plot=False, num_episodes=200,
            max_num_steps_per_episode=1000, learning_rate=0.001,
            memory_replay_size=10000, memory_policy_size=1000):
    """
    Soft Q-learning training routine. Retuns rewards and durations logs.
    Plot environment screen
    """
    num_envs = 4
    list_of_envs = []
    for i in range(num_envs):
        list_of_envs.append(gym.make('Breakout-v4'))

    # pi_0
    policy = Policy(list_of_envs[0], num_envs, alpha, beta)
    # Q value, every environment has one, used to calculate A_i,
    models = [DQN(list_of_envs[i], policy, alpha, beta, model_name="model_"+str(i)) for i in range(0, num_envs)]
    policy.add_models(models)

    # info list for each environment
    episode_durations = [[] for _ in range(num_envs)]   # list of local steps
    episode_rewards = [[] for _ in range(num_envs)]     # list of list of episode reward

    episodes_done = np.zeros(num_envs)      # episode num
    steps_done = np.zeros(num_envs)         # global timesteps for each env
    current_time = np.zeros(num_envs)       # local timesteps for each env
    episode_total_rewards = np.zeros(num_envs)
    # Initialize environments
    states = []
    for i in range(num_envs):
        states.append(policy.models[i].env.reset())

    while np.min(episodes_done) < num_episodes:
        #   1. do the step for each env
        for i in range(num_envs):
            action = policy.models[i].action(states[i])
            next_state, reward, done, _ = policy.models[i].env.step(action[0, 0])
            reward = [reward]
            episode_total_rewards[i] += reward

            if done:
                next_state = None

            steps_done[i] += 1      # global_steps
            current_time[i] += 1    # local steps
            time = [current_time[i]]
            # 把states scale [84, 84]
            policy.models[i].experience(states[i], action, next_state, reward, time)
            states[i] = next_state      # move to next state

            #   2. do one optimization step for each env using "soft-q-learning".
            # Perform one step of the optimization (on the target network)
            policy.models[i].optimizer_step()
            # ===========update step info end ========================

            # ===========update episode info begin ====================
            if done:
                print("ENV:", i, "iter:", episodes_done[i],
                    "\treward:", episode_total_rewards[i],
                    "\tit:", current_time[i], "\texp_factor:", eps_end +
                    (eps_start - eps_end) * math.exp(-1. * episodes_done[i] / eps_decay))

                states[i] = policy.models[i].env.reset()    # reset env
                episodes_done[i] += 1       # episode steps
                episode_durations[i].append(current_time[i])    # append each episode local timesteps list for every env
                current_time[i] = 0     # reset local timesteps
                episode_rewards[i].append(episode_total_rewards[i])     # append total episode_reward to list
                if is_plot:
                    plot_rewards(episode_rewards, i)
            # ===========update episode info end ====================

        #   3. do one optimization step for the policy
        # after all envs has performed one step, optimize policy
        policy.optimize_step()

    print('Complete')
    # env.render(close=True)
    # env.close()
    if is_plot:
        plt.ioff()
        plt.show()

    # Store Results
    np.save(file_name + '-distral-2col-rewards', episode_rewards)
    np.save(file_name + '-distral-2col-durations', episode_durations)

    return models, policy, episode_rewards, episode_durations
