import numpy as np
import os
import math
import pickle
import sys
import time
import queue
import multiprocessing as mp
from itertools import repeat
import tensorflow as tf
import threading
import  seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append(os.getcwd() + "/../")

import maddpg_.common.tf_util as U
from experiments.ops import time_begin, time_end, mkdir, make_env, get_trainers, sample_map
from experiments.uav_statistics import draw_util


def test(arglist, model_number):
  debug = False
  num_tasks = arglist.num_task  # 总共有多少个任务
  list_of_taskenv = []  # env list
  graph = tf.Graph()
  with graph.as_default():
    with U.single_threaded_session():
        if debug:
            begin = time_begin()
        # 1.1创建common actor
        env = make_env(arglist.scenario, reward_type=arglist.reward_type)
        env.set_map(sample_map(arglist.train_data_dir + arglist.train_data_name + "_1.h5"))
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        actors = get_trainers(env, "actor_", num_adversaries, obs_shape_n, arglist, type=0, lstm_scope="lstm")
        for i in range(num_tasks):
            list_of_taskenv.append(make_env(arglist.scenario, reward_type=arglist.reward_type))
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        
        # 1.2 Initialize
        U.initialize()

        arglist.load_dir = arglist.save_dir
        model_name = arglist.load_dir.split('/')[-2] + '/'
        path = arglist.pictures_dir_train_test + model_name
        mkdir(path)
        for i in range(num_tasks):
            mkdir(os.path.join(path, "task_" + str(i)))
        # 2.1 加载checkpoints
        model_load_dir = os.path.join(arglist.load_dir, str(model_number * arglist.save_rate), 'model.ckpt')
        print('From ', model_load_dir, ' Loading previous state...')
        U.load_state(model_load_dir)

        # 3.1 全局变量初始化
        global_steps = np.zeros(num_tasks)  # global timesteps for each env
        episodes_rewards = [[0.0] for _ in range(num_tasks)]  # 每个元素为在一个episode中所有agents rewards的和
        # agent_rewards[i]中的每个元素记录单个agent在一个episode中所有rewards的和
        agent_rewards = [[[0.0] for _ in range(env.n)] for _ in range(num_tasks)]

        energy_consumptions_for_test = [[] for _ in range(num_tasks)]
        j_index = [[] for _ in range(num_tasks)]
        aver_cover = [[] for _ in range(num_tasks)]
        instantaneous_dis = [[] for _ in range(num_tasks)]
        instantaneous_out_the_map = [[] for _ in range(num_tasks)]
        energy_efficiency = [[] for _ in range(num_tasks)]
        instantaneous_accmulated_reward = [[] for _ in range(num_tasks)]

        # 3.2 局部变量初始化
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

        # 3.3 初始化ENV
        obs_n_list = []
        for i in range(num_tasks):
            obs_n = list_of_taskenv[i].reset()
            list_of_taskenv[i].set_map(sample_map(arglist.train_data_dir + arglist.train_data_name + "_" + str(i+1) + ".h5"))
            obs_n_list.append(obs_n)

        # 3.4
        history_n = [[queue.Queue(arglist.history_length) for _ in range(env.n)] for _ in range(num_tasks)]
        for i in range(num_tasks):
            for j in range(env.n):
                for _ in range(arglist.history_length):
                    history_n[i][j].put(obs_n_list[i][j])
        # 4 test
        episode_start_time = time.time()
        print('Starting iterations...')
        episode_number = 0
        while True:
            for task_index in range(num_tasks):
                # 3.1更新环境
                current_env = list_of_taskenv[task_index]
                # get action
                action_n = [agent.action(obs) for agent, obs in zip(actors, history_n[task_index])]
                # environment step
                new_obs_n, rew_n, done_n, info_n = current_env.step(action_n)
                local_steps[task_index] += 1  # 更新局部计数器
                global_steps[task_index] += 1  # 更新全局计数器
                done = all(done_n)
                terminal = (local_steps[task_index] >= arglist.max_episode_len)
        
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
        
                episode_number = math.ceil(global_steps[task_index] / arglist.max_episode_len)
                if done or terminal:
                    # 记录每个episode的变量
                    energy_consumptions_for_test[task_index].append(energy_one_episode[task_index][-1])  # energy
                    j_index[task_index].append(j_index_one_episode[task_index][-1])  # fairness index
                    aver_cover[task_index].append(aver_cover_one_episode[task_index][-1])  # coverage
                    instantaneous_dis[task_index].append(
                        disconnected_number_one_episode[task_index][-1])  # disconnected
                    instantaneous_out_the_map[task_index].append(
                        over_map_one_episode[task_index][-1])  # out of the map
                    instantaneous_accmulated_reward[task_index].append(
                        accmulated_reward_one_episode[task_index][-1])  # reward
                    # energy_efficiency[task_index].append(aver_cover_one_episode[task_index][-1]
                    #                                      * j_index_one_episode[task_index][-1] /
                    #                                      energy_one_episode[task_index][-1])  # efficiency
                    energy_efficiency[task_index].append(j_index_one_episode[task_index][-1] /
                                                         energy_one_episode[task_index][-1])  # efficiency

                    episode_end_time = time.time()
                    episode_time = episode_end_time - episode_start_time
                    episode_start_time = episode_end_time

                    current_path = os.path.join(path, "task_" + str(task_index))
                    print('Task %d, Episode: %d - energy_consumptions: %s, efficiency: %s, time %s' % (
                        task_index,
                        episode_number,
                        str(current_env.get_energy_origin()),
                        str(energy_efficiency[task_index][-1]),
                        str(round(episode_time, 3))))
                    plt.figure()
                    _, (ax1, ax2) = plt.subplots(figsize=(22, 10), ncols=2)
                    sns.heatmap(current_env.map, annot=True, ax=ax1)
                    ax1.set_xlabel("target fair")
                    sns.heatmap(current_env.get_fair_matrix(), annot=True, ax=ax2)
                    ax2.set_xlabel("current fair")
                    plt.savefig(os.path.join(current_path,
                                "Model_" + str(model_number*arglist.save_rate) + "_Episode_" + str(episode_number) + "_fair.png"))
                    plt.close()
                    # 绘制reward曲线)
                    if arglist.draw_picture_test:
                        file_path = os.path.join(current_path,
                                                'test.log')
                        if episode_number == arglist.num_test_episodes:
                            report = '\nModel-' + str(model_number * arglist.save_rate) + \
                                     '-testing ' + str(arglist.num_test_episodes) + ' episodes\'s result:' \
                                     + '\n!!!Max energy efficiency: ' \
                                     + str(np.max(energy_efficiency[task_index])) \
                                     + '\n!!!Average energy efficiency: ' \
                                     + str(np.mean(energy_efficiency[task_index])) \
                                     + '\nAverage average attained coverage: ' \
                                     + str(np.mean(aver_cover[task_index])) + \
                                     '\nAverage Jaint\'s fairness index: ' \
                                     + str(np.mean(j_index[task_index])) + \
                                     '\nJaint\'s fairness index: ' \
                                     + str(j_index[task_index]) + \
                                     '\nAverage normalized average energy consumptions: ' \
                                     + str(np.mean(energy_consumptions_for_test[task_index])) \
                                     + "\n==========================end=============================\n"
                            draw_util.drawTest(model_number * arglist.save_rate,
                                               arglist.pictures_dir_train_test + model_name + "task_" + str(task_index) + "/",
                                               energy_efficiency[task_index],
                                               energy_consumptions_for_test[task_index],
                                               aver_cover[task_index],
                                               j_index[task_index],
                                               instantaneous_accmulated_reward[task_index],
                                               instantaneous_dis[task_index],
                                               instantaneous_out_the_map[task_index],
                                               len(aver_cover[task_index]),
                                               bl_coverage,
                                               bl_jainindex,
                                               bl_loss,
                                               False)
                            with open(file_path, 'a+') as file:
                                file.write(report)

                    # reset custom statistics variabl between episode and epoch------------------------------------
            
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
            
                    # 重置局部变量
                    obs_n_list[task_index] = current_env.reset()  # 重置env
                    current_env.set_map(
                        sample_map(arglist.train_data_dir + arglist.train_data_name + "_" + str(task_index + 1) + ".h5"))
                    local_steps[task_index] = 0  # 重置局部计数器
            
                    # 更新全局变量
                    episodes_rewards[task_index].append(0)  # 添加新的元素
                    for reward in agent_rewards[task_index]:
                        reward.append(0)
        
            if episode_number > arglist.num_test_episodes:
                break


def multi_process_time_calculate(arglist):
    total_model_number = int(100 / arglist.save_rate)
    begin_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(test, zip(repeat(arglist), range(1, total_model_number + 1)))
    total_time = time.time() - begin_time
    
    jobs = []
    begin_time = time.time()
    for model_number in range(1, total_model_number + 1):
        jobs.append(pool.apply_async(test, args=(arglist, model_number)))
    pool.close()
    pool.join()
    total_time2 = time.time() - begin_time
    
    jobs2 = []
    begin_time = time.time()
    for model_number in range(1, total_model_number + 1):
        jobs2.append(threading.Thread(target=test, args=(arglist, model_number)))
    for j in jobs2:
        j.start()
    for j in jobs2:
        j.join()
    total_time3 = time.time() - begin_time
    print("multi thread time: ", total_time)
    print("multi thread time: ", total_time2)
    print("multi thread time: ", total_time3)
    # pool.map: 638.7072842121124
    # pool.apply_asynctime: 567.7493116855621
    # threading.Thread time: 1165.6100940704346
    begin_time = time.time()
    for model_number in range(1, total_model_number + 1):
        test(arglist, model_number)
    print("single thread time: ", time.time() - begin_time)
    # single thread time: 1906.9262564182281
    print("Done")


def multi_process_test(arglist):
    total_model_number = int(arglist.max_test_model_number / arglist.save_rate)
    # pool.apply_async, multithread
    begin_time = time.time()
    jobs = []
    pool = mp.Pool(mp.cpu_count())
    for model_number in range(1, total_model_number+1):
        jobs.append(pool.apply_async(test, args=(arglist, model_number)))
    pool.close()
    pool.join()
    total_time = time.time() - begin_time
    print("multi thread time: ", total_time)
    print("Done")
