#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import time
import sys
import math
import pickle
import os
from multiprocessing import Pool

cwd = os.getcwd()
path = cwd + "/../"
sys.path.append(path)

import model.common.tf_util as U
from model.trainer.history import History
from experiments.uav_statistics import draw_util
from experiments.ops import make_env, get_trainers, sample_map


def time_begin():
        return time.time()


def time_end(begin_time, info):
        print(info)
        return time.time() - begin_time
        

def train(arglist):
        debug = False
        num_tasks = arglist.num_task  # 总共有多少个任务
        list_of_taskenv = []  # env list
        save_path = arglist.save_dir
        if not os.path.exists(save_path):
                os.makedirs(save_path)

        print("ok")
        with U.single_threaded_session():
                if debug:
                        begin = time_begin()
                # 1.初始化
                # 1.1创建一个actor
                env = make_env(arglist.scenario, arglist)
                env.set_map(sample_map(arglist.data_path + "_1.h5"))
                obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
                num_adversaries = min(env.n, arglist.num_adversaries)
                policy = get_trainers(env, "pi_0_", num_adversaries, obs_shape_n, arglist, is_actor=True, acotr=None)
                
                # 1.2创建每个任务的critic
                model_list = []         # 所有任务critic的list
                for i in range(num_tasks):
                        # 创建每个任务的env
                        list_of_taskenv.append(make_env(arglist.scenario, arglist))
                        trainers = get_trainers(list_of_taskenv[i], "task_"+str(i+1)+"_", num_adversaries,
                                                                                obs_shape_n,  arglist, is_actor=False, acotr=policy)
                        model_list.append(trainers)
                
                # 1.3 create p_train
                for task_index in range(num_tasks):
                        for actor, critic in zip(policy, model_list[task_index]):
                                actor.add_p(critic.name)
                                critic.p = actor.p_train
                        
                # 1.4 全局变量初始化
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
                
                global_steps_tensor = tf.Variable(tf.zeros(num_tasks), trainable=False)  # global timesteps for each env
                global_steps_ph = tf.placeholder(tf.float32, [num_tasks])
                global_steps_assign_op = tf.assign(global_steps_tensor, global_steps_ph)
                model_number = int(arglist.num_episodes / arglist.save_rate)
                saver = tf.train.Saver(max_to_keep=model_number)
                
                efficiency_list = []
                for i in range(num_tasks):
                        efficiency_list.append(tf.placeholder(tf.float32, shape=None, name="efficiency_placeholder"+str(i)))
                efficiency_summary_list = []
                for i in range(num_tasks):
                        efficiency_summary_list.append(tf.summary.scalar("efficiency_%s" % i, efficiency_list[i]))
                writer = tf.summary.FileWriter("../summary/efficiency")
                print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
                U.initialize()
                
                # 1.5 生成模型保存或者恢复文件夹目录
                if arglist.load_dir == "":
                        arglist.load_dir = save_path
                if arglist.display or arglist.restore or arglist.benchmark:
                        file_list = []
                        for f in os.listdir(arglist.load_dir):
                                if os.path.isdir(os.path.join(arglist.save_dir, f)):
                                        file_list.append(f)
                        file_list.sort(key=lambda fn: os.path.getmtime(arglist.load_dir + "/" + fn))
                        if len(file_list) > num_tasks:
                                load_dir = os.path.join(arglist.load_dir, file_list[-1], "model.ckpt")
                                U.load_state(load_dir)
                        print('Loading previous state...')
                
                global_steps = tf.get_default_session().run(global_steps_tensor)
                
                # 1.6 episode局部变量初始化
                local_steps = np.zeros(num_tasks)  # local timesteps for each env
                t_start = time.time()
                
                energy_one_episode = [[] for _ in range(num_tasks)]
                j_index_one_episode = [[] for _ in range(num_tasks)]
                aver_cover_one_episode = [[] for _ in range(num_tasks)]
                over_map_counter = np.zeros(num_tasks)
                over_map_one_episode = [[] for _ in range(num_tasks)]
                disconnected_number_counter = np.zeros(num_tasks)
                disconnected_number_one_episode = [[] for _ in range(num_tasks)]
                episode_reward_step = np.zeros(num_tasks)               # 累加一个episode里每一步的所有智能体的平均reward
                accmulated_reward_one_episode = [[] for _ in range(num_tasks)]
                route_one_episode = [[] for _ in range(num_tasks)]
                
                # 1.7 初始化ENV
                obs_n_list = []
                for i in range(num_tasks):
                        obs_n = list_of_taskenv[i].reset()
                        list_of_taskenv[i].set_map(sample_map(arglist.data_path + "_" + str(i+1) + ".h5"))
                        obs_n_list.append(obs_n)
                
                # 1.8 生成maddpg 加上rnn之后的输入seq，
                history_n = [[] for _ in range(num_tasks)]
                for i in range(num_tasks):
                        for j in range(len(obs_n_list[i])):  # 生成每个智能体长度为history_length的观测
                                history = History(arglist, [obs_shape_n[j][0]])
                                history_n[i].append(history)
                                for _ in range(arglist.history_length):
                                        history_n[i][j].add(obs_n_list[i][j])
                                
                if debug:
                        print(time_end(begin, "initialize"))
                        begin = time_begin()
                # 2.训练
                print('Starting iterations...')
                episode_start_time = time.time()
                state_dim = obs_shape_n[0][0]
                # 1.9
                # history_list = [tf.placeholder(tf.float32, shape=) for _ in range(env.n)]
                history_list = [[] for _ in range(env.n)]
                sess = tf.get_default_session()
                
                while True:
                        # for task_index in range(num_tasks):
                        #       action_n = []
                        #       # 用critic获得state,用critic给出action，
                        #       for idx, (agent, his) in enumerate(zip(policy, history_n[task_index])):
                        #               history_list[idx].append(his.obtain().reshape(1, state_dim, arglist.history_length))            # [1, state_dim, length]
                        #
                        # for idx in range(env.n):
                        #       hhh = np.concatenate(history_list[idx], 0)
                        #       temp_action = agent.action([hhh], [num_tasks])
                        #       action_n.append(temp_action)
                        # action_array = np.array(action_n)
                        # 2.1,在num_tasks个任务上进行采样
                        # action_n = action_array[:, task_index, :]
                        
                        for task_index in range(num_tasks):
                                action_n = []
                                # 用critic获得state,用critic给出action，
                                for agent, his in zip(policy, history_n[task_index]):
                                        hiss = his.obtain().reshape(1, state_dim, arglist.history_length)               # [1, state_dim, length]
                                        action = agent.action([hiss], [1])
                                        action_n.append(action[0])
                                # action_n = []
                                # # 用critic获得state,用critic给出action，
                                # results = []
                                # for agent, his in zip(policy, history_n[task_index]):
                                #       hiss = his.obtain().reshape(1, state_dim, arglist.history_length)  # [1, state_dim, length]
                                #       results.append(pool.apply_async(agent.action, args=([hiss], [1])))
                                # for action in results:
                                #       action_n.append(action.get())
                                # pool.close()
                                # pool.join()
                                
                                if debug:
                                        print(time_end(begin, "action2"))
                                        begin = time_begin()
                                current_actors = model_list[task_index]
                                current_env = list_of_taskenv[task_index]
                                # new_obs_n, rew_n, done_n = pool.apply(current_env, args=(action_n, ))
                                new_obs_n, rew_n, done_n = current_env.step(action_n)

                                if debug:
                                        print(time_end(begin, "env.step"))
                                        begin = time_begin()
                                
                                local_steps[task_index] += 1            # 更新局部计数器
                                global_steps[task_index] += 1           # 更新全局计数器
                                done = all(done_n)
                                terminal = (local_steps[task_index] >= arglist.max_episode_len)
                                # 收集experience
                                for i in range(env.n):
                                        current_actors[i].experience(obs_n_list[task_index][i], action_n[i], rew_n[i], done_n[i], terminal)
                                        policy[i].experience(obs_n_list[task_index][i], action_n[i], rew_n[i], done_n[i], terminal)
                                        
                                # 更新obs
                                obs_n_list[task_index] = new_obs_n
                                if debug:
                                        print(time_end(begin, "experience"))
                                        begin = time_begin()
                                # 2.2，优化每一个任务的critic
                                for i, rew in enumerate(rew_n):
                                        episodes_rewards[task_index][-1] += rew
                                        agent_rewards[task_index][i][-1] += rew
                                
                                for critic in current_actors:
                                        critic.preupdate()
                                for critic in current_actors:
                                        critic.update(current_actors, global_steps[task_index])
                                        
                                if debug:
                                        print(time_end(begin, "update critic"))
                                        begin = time_begin()
                                # 2.3，优化actor
                                # policy_step += 1
                                # print("policy steps: ", policy_step)
                                for actor, critic in zip(policy, current_actors):
                                        actor.change_p(critic.p)
                                        actor.update(policy, global_steps[task_index])
                                if debug:
                                        print(time_end(begin, "update actor"))
                                        begin = time_begin()
                                # 2.4 记录和更新train信息
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

                                if debug:
                                        print(time_end(begin, "others"))
                                        begin = time_begin()

                                episode_number = math.ceil(global_steps[task_index] / arglist.max_episode_len)
                                if done or terminal:
                                        model_name = save_path.split('/')[-2] + '/'
                                        temp_efficiency = np.array(aver_cover_one_episode[task_index]) * np.array(
                                                j_index_one_episode[task_index]) / np.array(energy_one_episode[task_index])
                                        draw_util.draw_single_episode(
                                                arglist.pictures_dir_train + model_name + "single_episode_task_" + str(task_index) + "/",
                                                episode_number,
                                                temp_efficiency,
                                                aver_cover_one_episode[task_index],
                                                j_index_one_episode[task_index],
                                                energy_one_episode[task_index],
                                                disconnected_number_one_episode[task_index],
                                                over_map_one_episode[task_index],
                                                accmulated_reward_one_episode[task_index]
                                        )
                                        # 记录每个episode的变量
                                        energy_consumptions_for_test[task_index].append(energy_one_episode[task_index][-1])             # energy
                                        j_index[task_index].append(j_index_one_episode[task_index][-1])         # fairness index
                                        aver_cover[task_index].append(aver_cover_one_episode[task_index][-1])           # coverage
                                        instantaneous_dis[task_index].append(disconnected_number_one_episode[task_index][-1])           # disconnected
                                        instantaneous_out_the_map[task_index].append(over_map_one_episode[task_index][-1])              # out of the map
                                        instantaneous_accmulated_reward[task_index].append(accmulated_reward_one_episode[task_index][-1])               # reward
                                        energy_efficiency[task_index].append(aver_cover_one_episode[task_index][-1]
                                                * j_index_one_episode[task_index][-1] / energy_one_episode[task_index][-1])             # efficiency

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
                                                feed_dict={efficiency_list[task_index]: energy_efficiency[task_index][-1]})
                                        writer.add_summary(efficiency_s, global_step=episode_number)

                                        # 应该在每个重置每个episode中的局部变量--------------------------------------------
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
                                                sample_map(arglist.data_path + "_" + str(task_index + 1) + ".h5"))
                                        local_steps[task_index] = 0  # 重置局部计数器
                                        
                                        # 更新全局变量
                                        episodes_rewards[task_index].append(0)  # 添加新的元素
                                        for reward in agent_rewards[task_index]:
                                                reward.append(0)
                                
                                # save model, display training output
                                if terminal and (episode_number % arglist.save_rate == 0):
                                        tf.get_default_session().run(global_steps_assign_op, feed_dict={global_steps_ph: global_steps})
                                        save_dir_custom = save_path + str(episode_number) + '/model.ckpt'
                                        U.save_state(save_dir_custom, saver=saver)
                                        # print statement depends on whether or not there are adversaries
                                        # 最新save_rate个episode的平均reward
                                        save_rate_mean_reward = np.mean(episodes_rewards[task_index][-arglist.save_rate:])
                                        if num_adversaries == 0:
                                                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                                                        global_steps[task_index], episode_number, save_rate_mean_reward,
                                                        round(time.time() - t_start, 3)))
                                        else:
                                                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                                                        global_steps[task_index], episode_number, save_rate_mean_reward,
                                                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards[task_index]], round(time.time() - t_start, 3)))
                                        
                                        t_start = time.time()
                                        
                                        final_ep_rewards[task_index].append(save_rate_mean_reward)
                                        for rew in agent_rewards[task_index]:
                                                final_ep_ag_rewards[task_index].append(np.mean(rew[-arglist.save_rate:]))
                                        
                                        # 保存train曲线
                                        if arglist.draw_picture_train:
                                                # model_name = save_path.split('/')[-2] + '/'
                                                draw_util.draw_episodes(
                                                        episode_number,
                                                        arglist.pictures_dir_train + model_name + "all_episodes_task_" + str(task_index) + "/",
                                                        aver_cover[task_index],
                                                        j_index[task_index],
                                                        energy_consumptions_for_test[task_index],
                                                        instantaneous_dis[task_index],
                                                        instantaneous_out_the_map[task_index],
                                                        energy_efficiency[task_index],
                                                        instantaneous_accmulated_reward[task_index],
                                                        len(aver_cover[task_index])
                                                )
                                # saves final episode reward for plotting training curve later
                                if episode_number > arglist.num_episodes:
                                        rew_file_name = arglist.plots_dir + arglist.exp_name + str(task_index) + '_rewards.pkl'
                                        with open(rew_file_name, 'wb') as fp:
                                                pickle.dump(final_ep_rewards, fp)
                                        agrew_file_name = arglist.plots_dir + arglist.exp_name + str(task_index) + '_agrewards.pkl'
                                        with open(agrew_file_name, 'wb') as fp:
                                                pickle.dump(final_ep_ag_rewards, fp)
                                                print('...Finished total of {} episodes.'.format(episode_number))
                        if episode_number > arglist.num_episodes:
                                break