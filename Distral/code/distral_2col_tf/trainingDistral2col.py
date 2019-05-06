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

    # Initialize environments
    for env in list_of_envs:
        state = env.reset()
        print(state.shape)


    # pi_0
    policy = Policy(list_of_envs[0], num_envs, alpha, beta)
    # Q value, every environment has one, used to calculate A_i,
    models = [DQN(list_of_envs[i], policy, alpha, beta, "model"+str(i)) for i in range(0, num_envs)]
    policy.add_models(models)

    # info list for each environment
    episode_durations = [[] for _ in range(num_envs)]   # list of local steps
    episode_rewards = [[] for _ in range(num_envs)]     # list of list of episode reward

    episodes_done = np.zeros(num_envs)      # episode num
    steps_done = np.zeros(num_envs)         # global timesteps for each env
    current_time = np.zeros(num_envs)       # local timesteps for each env


    while np.min(episodes_done) < num_episodes:
        #   1. do the step for each env
        for i in range(num_envs):
            # ===========update step info begin========================
            current_screen = get_screen(policy.models[i].env)
            # state
            state = current_screen # - last_screen
            # action chosen by pi_1~pi_i
            action = policy.models[i].action(state)

            # global_steps
            steps_done[i] += 1
            # local steps
            current_time[i] += 1
            # reward
            _, reward, done, _ = policy.models[i].env.step(action[0, 0])
            reward = [reward]

            # next state
            current_screen = get_screen(policy.models[i].env)
            if not done:
                next_state = current_screen # - last_screen
            else:
                next_state = None

            # add to buffer
            time = [current_time[i]]
            policy.models[i].experience(state, action, next_state, reward, time)
            #   2. do one optimization step for each env using "soft-q-learning".
            # Perform one step of the optimization (on the target network)
            policy.models[i].optimizer_step()
            # ===========update step info end ========================


            # ===========update episode info begin ====================
            if done:
                print("ENV:", i, "iter:", episodes_done[i],
                    "\treward:", policy.models[i].env.episode_total_reward,
                    "\tit:", current_time[i], "\texp_factor:", eps_end +
                    (eps_start - eps_end) * math.exp(-1. * episodes_done[i] / eps_decay))
                # reset env
                policy.models[i].env.reset()
                # episode steps
                episodes_done[i] += 1
                # append each episode local timesteps list for every env
                episode_durations[i].append(current_time[i])
                # reset local timesteps
                current_time[i] = 0
                # append total episode_reward to list
                episode_rewards[i].append(env.episode_total_reward)
                if is_plot:
                    plot_rewards(episode_rewards, i)
            # ===========update episode info end ====================

        #   3. do one optimization step for the policy
        # after all envs has performed one step, optimize policy
        policy.optimize_step()

    print('Complete')
    env.render(close=True)
    env.close()
    if is_plot:
        plt.ioff()
        plt.show()

    ## Store Results

    np.save(file_name + '-distral-2col-rewards', episode_rewards)
    np.save(file_name + '-distral-2col-durations', episode_durations)

    return models, policy, episode_rewards, episode_durations
