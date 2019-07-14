import numpy as np
import os
import tensorflow as tf
import math
import pickle
import sys
import time

sys.path.append(os.getcwd() + "/../")

import maddpg_.common.tf_util as U
from experiments.ops import time_begin, time_end, mkdir, make_env, get_trainers, sample_map
from experiments.uav_statistics import draw_util


def test(arglist):
    debug = False
    num_tasks = arglist.num_task  # 总共有多少个任务
    list_of_taskenv = []  # env list
    with U.single_threaded_session():
        if debug:
            begin = time_begin()
        # 1.1创建common actor
        env = make_env(arglist.scenario, arglist.benchmark)
        env.set_map(sample_map(arglist.test_data_dir + arglist.test_data_name + "_1.h5"))
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        actors = get_trainers(env, "actor_", num_adversaries, obs_shape_n, arglist, type=0)
        for i in range(num_tasks):
            list_of_taskenv.append(make_env(arglist.scenario))
            env.set_map(sample_map(arglist.test_data_dir + arglist.test_data_name + "_" + str(i) + ".h5"))
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        efficiency_list = []
        for i in range(num_tasks):
            efficiency_list.append(tf.placeholder(tf.float32, shape=None, name="efficiency_placeholder" + str(i)))
        efficiency_summary_list = []
        for i in range(num_tasks):
            efficiency_summary_list.append(tf.summary.scalar("efficiency_%s" % i, efficiency_list[i]))
        writer = tf.summary.FileWriter("../summary/efficiency")
        
        # 1.2 Initialize
        U.initialize()
        
        model_name = arglist.load_dir.split('/')[-2] + '/'
        mkdir(arglist.pictures_dir_test + model_name)
        model_index_step = 0
        model_number_total = arglist.num_train_episodes / arglist.save_rate
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
            
            # 2.1 加载checkpoints
            print('Loading previous state...')
            model_load_dir = os.path.join(arglist.load_dir, str(model_index_step * arglist.save_rate), 'model.ckpt')
            U.load_state(model_load_dir)

            # 2.2 全局变量初始化
            global_steps = np.zeros(num_tasks)  # global timesteps for each env
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

            # 2.3 局部变量初始化
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

            # 2.4 初始化ENV
            obs_n_list = []
            for i in range(num_tasks):
                obs_n = list_of_taskenv[i].reset()
                obs_n_list.append(obs_n)
                
            # 3 test
            episode_start_time = time.time()
            print('Starting iterations...')
            while True:
                for task_index in range(num_tasks):
                    # 3.1更新环境
                    current_env = list_of_taskenv[task_index]
                    # get action
                    action_n = [agent.action(obs) for agent, obs in zip(actors, obs_n)]
                    # environment step
                    new_obs_n, rew_n, done_n, info_n = current_env.step(action_n)
                    if debug:
                        print(time_end(begin, "env.step"))
                        begin = time_begin()
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
                        energy_efficiency[task_index].append(aver_cover_one_episode[task_index][-1]
                                                             * j_index_one_episode[task_index][-1] /
                                                             energy_one_episode[task_index][-1])  # efficiency

                        episode_end_time = time.time()
                        episode_time = episode_end_time - episode_start_time
                        episode_start_time = episode_end_time
                        print('Task %d, Episode: %d - energy_consumptions: %s, efficiency: %s, time %s' % (
                            task_index,
                            episode_number,
                            str(current_env.get_energy_origin()),
                            str(energy_efficiency[task_index][-1]),
                            str(round(episode_time, 3))))
                        
                        # 绘制reward曲线
                        efficiency_s = tf.get_default_session().run(efficiency_summary_list[task_index],
                                                                    feed_dict={efficiency_list[task_index]:
                                                                                   energy_efficiency[task_index][-1]})
                        
                        writer.add_summary(efficiency_s, global_step=episode_number)
                        if arglist.draw_picture_test:
                            if episode_number % arglist.save_rate == 0:
                                if np.mean(energy_efficiency) > max_average_energy_efficiency:
                                    max_model_index = model_index_step * arglist.save_rate - 1
                                    max_average_energy_efficiency = np.mean(energy_efficiency)
                                with open(arglist.pictures_dir_test + model_name + str(task_index) + 'test_report' + '.txt', 'a+') as file:
                                    report = '\nModel-' + str(model_index_step * arglist.save_rate - 1) + \
                                             '-testing ' + str(arglist.num_test_episodes) + ' episodes\'s result:' + \
                                             '\nAverage average attained coverage: ' + str(np.mean(aver_cover)) + \
                                             '\nAverage Jaint\'s fairness index: ' + str(np.mean(j_index)) + \
                                             '\nAverage normalized average energy consumptions:' + str(np.mean(energy_consumptions_for_test)) + \
                                             '\nAverage energy efficiency:' + str(np.mean(energy_efficiency)) + '\n'
                                    file.write(report)
                                draw_util.drawTest(model_index_step * arglist.save_rate - 1, arglist.pictures_dir_test + model_name,
                                                   energy_consumptions_for_test, aver_cover, j_index,
                                                   instantaneous_accmulated_reward, instantaneous_dis, instantaneous_out_the_map
                                                   , len(aver_cover), bl_coverage, bl_jainindex, bl_loss, energy_efficiency, False)
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
                            sample_map(arglist.test_data_dir + arglist.test_data_name + "_" + str(task_index + 1) + ".h5"))
                        local_steps[task_index] = 0  # 重置局部计数器

                        # 更新全局变量
                        episodes_rewards[task_index].append(0)  # 添加新的元素
                        for reward in agent_rewards[task_index]:
                            reward.append(0)

                    if episode_number > arglist.num_test_episodes:
                        mkdir(arglist.plots_dir)
                        rew_file_name = arglist.plots_dir + arglist.exp_name + str(task_index) + '_rewards.pkl'
                        with open(rew_file_name, 'wb') as fp:
                            pickle.dump(final_ep_rewards, fp)
                        agrew_file_name = arglist.plots_dir + arglist.exp_name + str(task_index) + '_agrewards.pkl'
                        with open(agrew_file_name, 'wb') as fp:
                            pickle.dump(final_ep_ag_rewards, fp)
                            print('...Finished total of {} episodes.'.format(episode_number))
                if episode_number > arglist.num_test_episodes:
                    break
