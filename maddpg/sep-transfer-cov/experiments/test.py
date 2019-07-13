import numpy as np
import os
import tensorflow as tf
import math
import pickle
import sys

sys.path.append(os.getcwd() + "/../")

import maddpg_.common.tf_util as U
from experiments.ops import time_begin, time_end, mkdir, make_env, get_trainers
from experiments.uav_statistics import draw_util


def test(arglist):
    debug = False
    num_tasks = arglist.num_task  # 总共有多少个任务
    list_of_taskenv = []  # env list
    load_path = arglist.load_dir
    with U.single_threaded_session():
        if debug:
            begin = time_begin()
        # 1.1创建每个任务的actor trainer和critic trainer
        trainers_list = []
        env = make_env(arglist.scenario, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        for i in range(num_tasks):
            list_of_taskenv.append(make_env(arglist.scenario))
            trainers = get_trainers(list_of_taskenv[i],
                                    "task_" + str(i + 1) + "_",
                                    num_adversaries,
                                    obs_shape_n,
                                    arglist)
            trainers_list.append(trainers)
    
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
    
        global_steps_tensor = tf.Variable(tf.zeros(num_tasks), trainable=False)  # global timesteps for each env
        global_steps_ph = tf.placeholder(tf.float32, [num_tasks])
        global_steps_assign_op = tf.assign(global_steps_tensor, global_steps_ph)
        model_number = int(arglist.num_episodes / arglist.save_rate)
        saver = tf.train.Saver(max_to_keep=model_number)
    
        efficiency_list = []
        for i in range(num_tasks):
            efficiency_list.append(tf.placeholder(tf.float32, shape=None, name="efficiency_placeholder" + str(i)))
        efficiency_summary_list = []
        for i in range(num_tasks):
            efficiency_summary_list.append(tf.summary.scalar("efficiency_%s" % i, efficiency_list[i]))
        writer = tf.summary.FileWriter("../summary/efficiency")
    
        # Initialize
        U.initialize()
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var)

        if debug:
            print(time_end(begin, "initialize"))
            begin = time_begin()
            
        model_name = arglist.load_dir.split('/')[-2] + '/'
        mkdir(arglist.pictures_dir_test + model_name)
        model_index_step = 0
        model_number_total = arglist.train_num_episodes / arglist.save_rate
        max_model_index = 0
        max_average_energy_efficiency = 0

        while True:
            if model_index_step >= model_number_total:
                with open(arglist.pictures_dir_test + model_name + 'test_report' + '.txt', 'a+') as file:
                    report = '\nModel ' + str(max_model_index) + ' attained max average energy efficiency' + \
                             '\nMax average energy efficiency:' + str(max_average_energy_efficiency)
                    file.write(report)
                break
            else:
                model_index_step += 1
            
            # 1.4 加载checkpoints
            if arglist.load_dir == "":
                arglist.load_dir = arglist.save_dir
            if arglist.display or arglist.restore or arglist.benchmark:
                print('Loading previous state...')
                model_load_dir = arglist.load_dir + str(model_index_step * arglist.save_rate - 1) + '/'
                U.load_state(arglist.load_dir)
            # global_steps = tf.get_default_session().run(global_steps_tensor)

            # 1.5 初始化ENV
            obs_n_list = []
            for i in range(num_tasks):
                obs_n = list_of_taskenv[i].reset()
                obs_n_list.append(obs_n)

            # 1.2 全局变量初始化
            episodes_rewards = [[0.0] for _ in range(num_tasks)]  # 每个元素为在一个episode中所有agents rewards的和
            # agent_rewards[i]中的每个元素记录单个agent在一个episode中所有rewards的和
            agent_rewards = [[[0.0] for _ in range(env.n)] for _ in range(num_tasks)]
            final_ep_rewards = [[] for _ in range(num_tasks)]  # sum of rewards for training curve
            final_ep_ag_rewards = [[] for _ in range(num_tasks)]  # agent rewards for training curve

            energy_consumptions_for_test = [[] for _ in range(num_tasks)]
            j_index = [[] for _ in range(num_tasks)]
            aver_cover = [[] for _ in range(num_tasks)]
            instantaneous_dis = [[] for _ in range(num_tasks)]
            instantaneous_out_the_map = [[] for _ in range(num_tasks)]
            energy_efficiency = [[] for _ in range(num_tasks)]
            instantaneous_accmulated_reward = [[] for _ in range(num_tasks)]

            # 1.3 局部变量初始化
            local_steps = np.zeros(num_tasks)  # local timesteps for each env
            energy_one_episode = [[] for _ in range(num_tasks)]
            j_index_one_episode = [[] for _ in range(num_tasks)]
            aver_cover_one_episode = [[] for _ in range(num_tasks)]
            over_map_counter = np.zeros(num_tasks)
            over_map_one_episode = [[] for _ in range(num_tasks)]
            disconnected_number_counter = np.zeros(num_tasks)
            disconnected_number_one_episode = [[] for _ in range(num_tasks)]
            episode_reward_step = np.zeros(num_tasks)  # 累加一个episode里每一步的所有智能体的平均reward
            accmulated_reward_one_episode = [[] for _ in range(num_tasks)]
            route_one_episode = [[] for _ in range(num_tasks)]
            

            bl_coverage = 0.8
            bl_jainindex = 0.8
            bl_loss = 100
            energy_efficiency = []

            print('Starting iterations...')
            while True:
                for task_index in range(num_tasks):
                    # 2.1更新环境，采集样本
                    current_env = list_of_taskenv[task_index]
                    current_trainers = trainers_list[task_index]
                    # get action
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                    # environment step
                    new_obs_n, rew_n, done_n, info_n = current_env.step(action_n)
                    if debug:
                        print(time_end(begin, "env.step"))
                        begin = time_begin()
                    local_steps[task_index] += 1  # 更新局部计数器
                    global_steps[task_index] += 1  # 更新全局计数器
                    done = all(done_n)
                    terminal = (local_steps[task_index] >= arglist.max_episode_len)
                    # 收集experience
                    for i in range(env.n):
                        current_trainers[i].experience(obs_n_list[task_index][i], action_n[i], rew_n[i], new_obs_n[i],
                                                       done_n[i], terminal)

                    # 更新obs
                    obs_n_list[task_index] = new_obs_n
                    # 更新reward
                    for i, rew in enumerate(rew_n):
                        episodes_rewards[task_index][-1] += rew
                        agent_rewards[task_index][i][-1] += rew
                    # energy
                    energy_one_episode[task_index].append(current_env.get_energy())
                    # fair index
                    j_index_one_episode[task_index].append(current_env.get_jain_index())
                    # coverage
                    aver_cover_one_episode[task_index].append(current_env.get_aver_cover())
                    # over map counter
                    over_map_counter[task_index] += current_env.get_over_map()
                    over_map_one_episode[task_index].append(over_map_counter[task_index])
                    # disconnected counter
                    disconnected_number_counter[task_index] += current_env.get_dis()
                    disconnected_number_one_episode[task_index].append(disconnected_number_counter[task_index])
                    # reward
                    episode_reward_step[task_index] += np.mean(rew_n)
                    accmulated_reward_one_episode[task_index].append(episode_reward_step[task_index])
                    route = current_env.get_agent_pos()
                    route_one_episode[task_index].append(route)

                    if done or terminal:
                        # reset custom statistics variabl between episode and epoch---------------------------------------------
                        instantaneous_accmulated_reward.append(accmulated_reward_one_episode[-1])
                        j_index.append(j_index_one_episode[-1])
                        instantaneous_dis.append(disconnected_number_one_episode[-1])
                        instantaneous_out_the_map.append(over_map_one_episode[-1])
                        aver_cover.append(aver_cover_one_episode[-1])
                        energy_consumptions_for_test.append(energy_one_episode[-1])
                        energy_efficiency.append(aver_cover_one_episode[-1] * j_index_one_episode[-1] / energy_one_episode[-1])
                        print('Episode: %d - energy_consumptions: %s ' % (train_step / arglist.max_episode_len,
                                                                        str(env._get_energy_origin())))

                        if task_index == num_tasks - 1:
                            energy_one_episode = [[] for _ in range(num_tasks)]
                            j_index_one_episode = [[] for _ in range(num_tasks)]
                            aver_cover_one_episode = [[] for _ in range(num_tasks)]
                            over_map_counter = np.zeros(num_tasks)
                            over_map_one_episode = [[] for _ in range(num_tasks)]
                            disconnected_number_counter = np.zeros(num_tasks)
                            disconnected_number_one_episode = [[] for _ in range(num_tasks)]
                            episode_reward_step = np.zeros(num_tasks)
                            accmulated_reward_one_episode = [[] for _ in range(num_tasks)]
                            route_one_episode = [[] for _ in range(num_tasks)]

                        if arglist.draw_picture_test:
                            if len(episode_rewards) % arglist.save_rate == 0:
                                if np.mean(energy_efficiency) > max_average_energy_efficiency:
                                    max_model_index = model_index_step * arglist.save_rate - 1
                                    max_average_energy_efficiency = np.mean(energy_efficiency)
                                with open(arglist.pictures_dir_test + model_name + 'test_report' + '.txt', 'a+') as file:
                                    report = '\nModel-' + str(model_index_step * arglist.save_rate - 1) + \
                                             '-testing ' + str(arglist.num_episodes) + ' episodes\'s result:' + \
                                             '\nAverage average attained coverage: ' + str(np.mean(aver_cover)) + \
                                             '\nAverage Jaint\'s fairness index: ' + str(np.mean(j_index)) + \
                                             '\nAverage normalized average energy consumptions:' + str(np.mean(energy_consumptions_for_test)) + \
                                             '\nAverage energy efficiency:' + str(np.mean(energy_efficiency)) + '\n'
                                    file.write(report)
                                draw_util.drawTest(model_index_step * arglist.save_rate - 1, arglist.pictures_dir_test + model_name,
                                                   energy_consumptions_for_test, aver_cover, j_index,
                                                   instantaneous_accmulated_reward, instantaneous_dis, instantaneous_out_the_map
                                                   , len(aver_cover), bl_coverage, bl_jainindex, bl_loss, energy_efficiency, False)
                        # reset custom statistics variabl between episode and epoch----------------------------------------

                    # for displaying learned policies
                    if arglist.draw_picture_test:
                        if len(episode_rewards) > arglist.num_episodes:
                            break
                        continue

                    # saves final episode reward for plotting training curve later
                    if len(episode_rewards) > arglist.num_episodes:
                        rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                        with open(rew_file_name, 'wb') as fp:
                            pickle.dump(final_ep_rewards, fp)
                        agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                        with open(agrew_file_name, 'wb') as fp:
                            pickle.dump(final_ep_ag_rewards, fp)
                        print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                        break
