import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch
import math
import numpy as np
from memory_replay import ReplayMemory, Transition
from model import DQN
from policy import Policy
from network import select_action, optimize_model, optimize_policy
import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
from utils import plot_rewards, plot_durations, plot_state, get_screen

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
    # action dimension
    num_actions = list_of_envs[0].action_space.n
    # total envs
    num_envs = len(list_of_envs)
    # pi_0
    policy = Policy(num_actions)
    # Q value, every environment has one, used to calculate A_i,
    models = [DQN(num_actions) for _ in range(0, num_envs)]   ### Add torch.nn.ModuleList (?)
    # replay buffer for env ???
    memories = [ReplayMemory(memory_replay_size, memory_policy_size) for _ in range(0, num_envs)]

    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    # optimizer for every Q model
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate)
                    for model in models]
    # optimizer for policy
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), )

    # info list for each environment
    episode_durations = [[] for _ in range(num_envs)]   # list of local steps
    episode_rewards = [[] for _ in range(num_envs)]     # list of list of episode reward

    episodes_done = np.zeros(num_envs)      # episode num
    steps_done = np.zeros(num_envs)         # global timesteps for each env
    current_time = np.zeros(num_envs)       # local timesteps for each env

    # Initialize environments
    for env in list_of_envs:
        env.reset()

    while np.min(episodes_done) < num_episodes:
        # TODO: add max_num_steps_per_episode

        # Optimization is given by alterating minimization scheme:
        #   1. do the step for each env
        #   2. do one optimization step for each env using "soft-q-learning".
        #   3. do one optimization step for the policy

        #   1. do the step for each env
        for i_env, env in enumerate(list_of_envs):
            # ===========update step info begin========================
            current_screen = get_screen(env)
            # state
            state = current_screen # - last_screen
            # action chosen by pi_1~pi_i
            action = select_action(state, policy, models[i_env], num_actions,
                                    eps_start, eps_end, eps_decay,
                                    episodes_done[i_env], alpha, beta)
            # global_steps
            steps_done[i_env] += 1
            # local steps
            current_time[i_env] += 1
            # reward
            _, reward, done, _ = env.step(action[0, 0])
            reward = [reward]

            # next state
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen # - last_screen
            else:
                next_state = None

            # add to buffer
            time = [current_time[i_env]]
            memories[i_env].push(state, action, next_state, reward, time)

            #   2. do one optimization step for each env using "soft-q-learning".
            # Perform one step of the optimization (on the target network)
            optimize_model(policy, models[i_env], optimizers[i_env],
                            memories[i_env], batch_size, alpha, beta, gamma)
            # ===========update step info end ========================


            # ===========update episode info begin ====================
            if done:
                print("ENV:", i_env, "iter:", episodes_done[i_env],
                    "\treward:", env.episode_total_reward,
                    "\tit:", current_time[i_env], "\texp_factor:", eps_end +
                    (eps_start - eps_end) * math.exp(-1. * episodes_done[i_env] / eps_decay))
                # reset env
                env.reset()
                # episode steps
                episodes_done[i_env] += 1
                # append each episode local timesteps list for every env
                episode_durations[i_env].append(current_time[i_env])
                # reset local timesteps
                current_time[i_env] = 0
                # append total episode_reward to list
                episode_rewards[i_env].append(env.episode_total_reward)
                if is_plot:
                    plot_rewards(episode_rewards, i_env)
            # ===========update episode info end ====================

        #   3. do one optimization step for the policy
        # after all envs has performed one step, optimize policy
        optimize_policy(policy, policy_optimizer, memories, batch_size,
                    num_envs, gamma)

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
