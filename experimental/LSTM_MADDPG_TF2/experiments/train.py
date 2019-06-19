#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import h5py
import sys
import os
sys.path.append("/home/mxxmhh/mxxhcm/code/")

import experimental.LSTM_MADDPG_TF2.model.common.tf_util as U
from experimental.LSTM_MADDPG_TF2.model.trainer.history import History
from experimental.LSTM_MADDPG_TF2.experiments.uav_statistics import draw_util
from experimental.LSTM_MADDPG_TF2.multiagent.uav.flag import FLAGS
from experimental.LSTM_MADDPG_TF2.experiments.ops import make_env, get_trainers, sample_map


def time_begin():
	return time.time()


def time_end(begin_time, info):
	print(info)
	return time.time() - begin_time
	
def train(arglist):
	with U.single_threaded_session():
		save_path = arglist.save_dir + "/"
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		
		# 1.初始化
		num_tasks = arglist.num_task		# 总共有多少个任务
		list_of_taskenv = []		# env list

		# 1.1创建一个actor
		env = make_env(arglist.scenario, arglist, arglist.benchmark)
		env.set_map(sample_map(arglist.data_path + "_1.h5"))
		obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
		num_adversaries = min(env.n, arglist.num_adversaries)
		policy = get_trainers(env, "pi_0_", num_adversaries, obs_shape_n, arglist, is_actor=True, acotr=None)
		
		# 1.2创建每个任务的critic
		model_list = []		# 所有任务critic的list
		for i in range(num_tasks):
			# 创建每个任务的env
			list_of_taskenv.append(make_env(arglist.scenario, arglist, arglist.benchmark))
			trainers = get_trainers(list_of_taskenv[i], "task_"+str(i+1)+"_", num_adversaries,
										obs_shape_n,  arglist, is_actor=False, acotr=policy)
			model_list.append(trainers)
		
		# for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
		# 	print(var)
		# 1.4全局变量初始化
		global_steps = tf.Variable(tf.zeros(num_tasks), trainable=False) # global timesteps for each env
		global_steps_add_op = []
		for i in range(num_tasks):
			global_steps_add_op.append(global_steps[i].assign(global_steps[i]+1))
		model_number = int(arglist.num_episodes / arglist.save_rate)
		saver = tf.train.Saver(max_to_keep=model_number)
		
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
		
		print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
		U.initialize()
		
		# 1.3生成模型保存或者恢复文件夹目录
		if arglist.load_dir == "":
			arglist.load_dir = save_path
		if arglist.display or arglist.restore or arglist.benchmark:
			file_list = []
			for f in os.listdir(arglist.load_dir):
				if os.path.isdir(os.path.join(arglist.save_dir, f)):
					file_list.append(f)
			file_list.sort(key=lambda fn: os.path.getmtime(arglist.load_dir + "/" + fn))
			if len(file_list) > 0:
				load_dir = os.path.join(arglist.load_dir, file_list[0], "model.ckpt")
				U.load_state(load_dir)
			print('Loading previous state...')
		
		
		# 1.5 episode局部变量初始化
		local_steps = np.zeros(num_tasks)  # local timesteps for each env
		t_start = time.time()
		
		energy_one_episode = [[] for _ in range(num_tasks)]
		j_index_one_episode = [[] for _ in range(num_tasks)]
		aver_cover_one_episode = [[] for _ in range(num_tasks)]
		over_map_counter = np.zeros(num_tasks)
		over_map_one_episode = [[] for _ in range(num_tasks)]
		disconnected_number_counter = np.zeros(num_tasks)
		disconnected_number_one_episode = [[] for _ in range(num_tasks)]
		episode_reward_step = np.zeros(num_tasks)		# 累加一个episode里每一步的所有智能体的平均reward
		accmulated_reward_one_episode = [[] for _ in range(num_tasks)]
		route = [[] for _ in range(num_tasks)]
		
		
		# 1.6初始化ENV
		obs_n_list = []
		for i in range(num_tasks):
			obs_n = list_of_taskenv[i].reset()
			list_of_taskenv[i].set_map(sample_map(arglist.data_path + "_" + str(i+1) + ".h5"))
			obs_n_list.append(obs_n)
		
		# 1.7 生成maddpg 加上rnn之后的输入seq，
		history_n = [[] for _ in range(num_tasks)]
		for i in range(num_tasks):
			for j in range(len(obs_n_list[i])):  # 生成每个智能体长度为history_length的观测
				history = History(arglist, [obs_shape_n[j][0]])
				history_n[i].append(history)
				for _ in range(arglist.history_length):
					history_n[i][j].add(obs_n_list[i][j])
		
		# 1.8 create p_train
		for task_index in range(num_tasks):
			for actor, critic in zip(policy, model_list[task_index]):
				actor.add_p(critic.name)
				critic.p = actor.p_train
				
		# 1.9 reward figures
		figures = [plt.figure() for _ in range(num_tasks)]
		axes = []
		for fig in figures:
			axes.append(fig.gca())
		
		# print(time_end(begin, "initialize"))
		# begin = time_begin()
		# 2.训练
		print('Starting iterations...')
		episode_start_time = time.time()
		while True:
			# print(time_end(begin, "loop"))
			# begin = time_begin()
			# 2.1,在num_tasks个任务上进行采样
			for task_index in range(num_tasks):
				current_env = list_of_taskenv[task_index]
				action_n = []
				# 用critic获得state,用critic给出action，
				# print(time_end(begin, "begin"))
				# begin = time_begin()
				for agent, his in zip(policy, history_n[task_index]):
					hiss = his.obtain().reshape(1, obs_shape_n[0][0], arglist.history_length)		# [1, state_dim, length]
					action = agent.action([hiss], [1])
					action_n.append(action)
				# print(time_end(begin, "action"))
				# begin = time_begin()
				# environment step
				# begin = time.time()
				new_obs_n, rew_n, done_n = current_env.step(action_n)
				# print(time.time() - begin)
				# print(time_end(begin, "env.step"))
				# begin = time_begin()
				
				local_steps[task_index] += 1		# 更新局部计数器
				step = tf.get_default_session().run(global_steps_add_op[task_index])
				# global_steps[task_index] += 1		# 更新全局计数器
				
				done = all(done_n)
				terminal = (local_steps[task_index] >= arglist.max_episode_len)
				# print(time_end(begin, "update counter"))
				# begin = time_begin()
				# 收集experience
				for i, agent in enumerate(model_list[task_index]):
					agent.experience(obs_n_list[task_index][i], action_n[i], rew_n[i], done_n[i], terminal)
				# 更新obs
				# obs_n_list[task_index] = new_obs_n
				# print(time_end(begin, "experience"))
				# begin = time_begin()
				# 2.2，优化每一个任务的critic
				for i, rew in enumerate(rew_n):
					episodes_rewards[task_index][-1] += rew
					agent_rewards[task_index][i][-1] += rew
				
				for critic in model_list[task_index]:
					critic.preupdate()
				for critic in model_list[task_index]:
					critic.update(model_list[task_index], step)
				# print(time_end(begin, "update critic"))
				# begin = time_begin()
				# 2.3，优化actor
				# policy_step += 1
				# print("policy steps: ", policy_step)
				for actor, critic in zip(policy, model_list[task_index]):
					actor.change_p(critic.p)
					actor.update(policy, step)
				# print(time_end(begin, "update actor"))
				# begin = time_begin()
				# 2.4 记录和更新train信息
				# - energy
				energy_one_episode[task_index].append(current_env.get_energy())
				# - fair index
				j_index_one_episode[task_index].append(current_env.get_jain_index())
				# - coverage
				aver_cover_one_episode[task_index].append(current_env.get_aver_cover())
				# - over map counter
				over_map_counter[task_index] += current_env.get_over_map()
				over_map_one_episode[task_index].append(over_map_counter[task_index])
				# - disconnected counter
				disconnected_number_counter[task_index] += current_env.get_dis()
				disconnected_number_one_episode[task_index].append(disconnected_number_counter)
				# - reward
				episode_reward_step[task_index] += np.mean(rew_n)
				accmulated_reward_one_episode[task_index].append(episode_reward_step)
				# - state
				s_route = current_env.get_agent_pos()
				for route_i in range(0, FLAGS.num_uav * 2, 2):
					tmp = [s_route[route_i], s_route[route_i + 1]]
					route.append(tmp)
				
				# print(time_end(begin, "task_index: " + str(task_index)+", steps:" + str(global_steps[task_index])))
				# begin = time_begin()
				
				episode_number = len(episodes_rewards[task_index])
				if done or terminal:
					# 记录每个episode的变量
					# - energy
					energy_consumptions_for_test[task_index].append(energy_one_episode[task_index][-1])
					# - fairness index
					j_index[task_index].append(j_index_one_episode[task_index][-1])
					# - coverage
					aver_cover[task_index].append(aver_cover_one_episode[task_index][-1])
					# - disconnected
					instantaneous_dis[task_index].append(disconnected_number_one_episode[task_index][-1])
					# - out of the map
					instantaneous_out_the_map[task_index].append(over_map_one_episode[task_index][-1])
					# - reward
					instantaneous_accmulated_reward[task_index].append(accmulated_reward_one_episode[task_index][-1])
					# - efficiency
					energy_efficiency[task_index].append(
						aver_cover_one_episode[task_index][-1] * j_index_one_episode[task_index][-1] / energy_one_episode[task_index][-1])
					episode_end_time = time.time()
					episode_time = episode_end_time - episode_start_time
					episode_start_time = episode_end_time
					# print(str(policy_step), " step time: ", round(step_time, 3))
					print(
						'Task %d, '
						'episode: %d, - '
						'energy_consumptions: %s,'
						'energy_efficiency : %s,'
						'time : %s.' %
						(
							task_index,
							episode_number,
							str(current_env.get_energy_origin()),
							str(energy_efficiency[task_index][-1]),
							str(round(episode_time, 3))
						)
					)
					
					# 绘制reward曲线
					plt.ion()
					axes[task_index].plot(np.arange(0, episode_number), energy_efficiency[task_index])
					plt.xlabel("episode number")
					plt.ylabel("energy efficiency")
					plt.savefig(save_path + str(task_index) + "efficiency.png")
					plt.ioff()
					
					# 重置每个episode中的局部变量--------------------------------------------
					energy_one_episode = [[] for _ in range(num_tasks)]
					j_index_one_episode = [[] for _ in range(num_tasks)]
					aver_cover_one_episode = [[] for _ in range(num_tasks)]
					over_map_counter = np.zeros(num_tasks)
					over_map_one_episode = [[] for _ in range(num_tasks)]
					disconnected_number_counter = np.zeros(num_tasks)
					disconnected_number_one_episode = [[] for _ in range(num_tasks)]
					episode_reward_step = np.zeros(num_tasks)
					accmulated_reward_one_episode = [[] for _ in range(num_tasks)]
					route = [[] for _ in range(num_tasks)]
					
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
					save_dir_custom = save_path + str(episode_number) + '/model.ckpt'
					print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
					U.save_state(save_dir_custom, saver=saver)
					# print statement depends on whether or not there are adversaries
					# 最新save_rate个episode的平均reward
					save_rate_mean_reward = np.mean(episodes_rewards[task_index][-arglist.save_rate:])
					if num_adversaries == 0:
						print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
							step, episode_number, save_rate_mean_reward,
							round(time.time() - t_start, 3)))
					else:
						print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
							step, episode_number, save_rate_mean_reward,
							[np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards[task_index]], round(time.time() - t_start, 3)))
					
					t_start = time.time()
					
					final_ep_rewards[task_index].append(save_rate_mean_reward)
					for rew in agent_rewards[task_index]:
						final_ep_ag_rewards[task_index].append(np.mean(rew[-arglist.save_rate:]))
					
					# 保存train曲线
					if arglist.draw_picture_train:
						model_name = save_path.split('/')[-2] + '/'
						draw_util.draw_episode(
							episode_number,
							arglist.pictures_dir_train + model_name + str(task_index) + "/",
							aver_cover[task_index],
							j_index[task_index],
							instantaneous_accmulated_reward[task_index],
							instantaneous_dis[task_index],
							instantaneous_out_the_map[task_index],
							len(aver_cover[task_index])
						)

				# saves final episode reward for plotting training curve later
				if episode_number > arglist.num_episodes:
					print('...Finished total of {} episodes.'.format(episode_number))
					break
