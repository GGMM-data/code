#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import h5py

import experimental.LSTM_MADDPG_TF2.model.common.tf_util as U
from experimental.LSTM_MADDPG_TF2.model.trainer.history import History
from experimental.LSTM_MADDPG_TF2.experiments.uav_statistics import draw_util
from experimental.LSTM_MADDPG_TF2.multiagent.uav.flag import FLAGS
from experimental.LSTM_MADDPG_TF2.experiments.ops import make_env, get_trainers, sample_map


def parse_args():
	parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

	parser.add_argument("--gamma", type=float, default=0.80, help="discount factor")
	parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
	parser.add_argument("--num-units", type=int, default=160, help="number of units in the mlp")
	parser.add_argument("--buffer-size", type=int, default=100, help="buffer capacity")
	parser.add_argument("--num-task", type=int, default=3, help="number of tasks")
	# rnn 长度
	parser.add_argument('--history_length', type=int, default=4, help="how many history states were used")
	parser.add_argument("--model-dir", type=str, default="./tmp/policy_gamma_0.80_batch_1024_neural_160_batch_75/",
						help="directory in which training state and model should be saved")
	
	# Environment
	parser.add_argument("--scenario", type=str, default="simple_uav", help="name of the scenario script")
	parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
	parser.add_argument("--num-episodes", type=int, default=4000, help="number of episodes")
	parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
	parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
	parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
	
	# Core training parameters
	parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
	# Checkpoint
	parser.add_argument("--exp-name", type=str, default="simple_uav", help="name of the experiment")
	parser.add_argument("--save-rate", type=int, default=100,
						help="save model once every time this many episodes are completed")
	parser.add_argument("--load-dir", type=str, default="./tmp/policy_f_1_u_7_r_3_c_5_with_wall/2599/",
						help="directory in which training state and model are loaded")
	# Evaluation
	parser.add_argument("--restore", action="store_true", default=False)
	parser.add_argument("--display", action="store_true", default=False)
	parser.add_argument("--benchmark", action="store_true", default=False)
	parser.add_argument("--draw-picture-train", action="store_true", default=True)
	parser.add_argument("--draw-picture-test", action="store_true", default=False)
	parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
	parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
						help="directory where benchmark data is saved")
	parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
						help="directory where plot data is saved")
	parser.add_argument("--pictures-dir-train", type=str, default="./result_pictures/train/",
						help="directory where result pictures data is saved")
	parser.add_argument("--pictures-dir-test", type=str, default="./result_pictures/test/",
						help="directory where result pictures data is saved")
	
	parser.add_argument("--cnn-format", type=str, default='NHWC', help="cnn_format")
	
	# custom parameters for uav
	return parser.parse_args()


def train(arglist):
	with U.single_threaded_session():
		# 1.初始化
		num_tasks = arglist.num_task		# 总共有多少个任务
		list_of_taskenv = []		# env list

		# 1.1创建一个actor
		env = make_env(arglist.scenario, arglist, arglist.benchmark)
		env.set_map(sample_map("../data/chengdu_1.h5"))
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
		for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
			print(var)
		print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
		U.initialize()
		
		# 1.3生成模型保存或者恢复文件夹目录
		if arglist.load_dir == "":
			arglist.load_dir = arglist.save_dir
		if arglist.display or arglist.restore or arglist.benchmark:
			print('Loading previous state...')
			U.load_state(arglist.load_dir)
		
		# 1.4全局变量初始化
		global_steps = np.zeros(num_tasks)  # global timesteps for each env
		policy_step = 0		# 记录actor训练的步数
		model_number = int(arglist.num_episodes / arglist.save_rate)
		saver = tf.train.Saver(max_to_keep=model_number)
		episodes_rewards = [0.0]  # 每个元素为在一个episode中所有agents rewards的和
		agent_rewards = [[0.0] for _ in range(env.n)]  # agent_rewards[i]中的每个元素记录单个agent在一个episode中所有rewards的和
		final_ep_rewards = []  # sum of rewards for training curve
		final_ep_rewards_sum = tf.Variable([0.0], name="final_ep_rewards")
		final_ep_rewards_sum_op = tf.summary.scalar('final_ep_r', final_ep_rewards_sum)
		final_ep_rewards_sum_ph = tf.placeholder(tf.float32, shape=None)
		final_ep_rewards_sum_assign_op = final_ep_rewards_sum.assign(final_ep_rewards_sum_ph)
		
		final_ep_ag_rewards = []  # agent rewards for training curve
		
		energy_consumptions_for_test = []
		j_index = []
		aver_cover = []
		instantaneous_dis = []
		instantaneous_out_the_map = []
		energy_efficiency = []
		instantaneous_accmulated_reward = []
		
		# 1.5 episode局部变量初始化
		local_steps = np.zeros(num_tasks)  # local timesteps for each env
		t_start = time.time()
		
		energy_one_episode = []
		j_index_one_episode = []
		aver_cover_one_episode = []
		over_map_counter = 0
		over_map_one_episode = []
		disconnected_number_counter = 0
		disconnected_number_one_episode = []
		episode_reward_step = 0		# 累加一个episode里每一步的所有智能体的平均reward
		accmulated_reward_one_episode = []
		route = []
		
		
		# 1.6初始化ENV
		obs_n_list = []
		for i in range(num_tasks):
			obs_n = list_of_taskenv[i].reset()
			list_of_taskenv[i].set_map(sample_map("../data/chengdu_" + str(i+1) + ".h5"))
			obs_n_list.append(obs_n)
		
		# 1.7 生成maddpg 加上rnn之后的输入seq，
		history_n = [[] for _ in range(num_tasks)]
		for i in range(num_tasks):
			for j in range(len(obs_n_list[i])):  # 生成每个智能体长度为history_length的观测
				history = History(arglist, [obs_shape_n[j][0]])
				history_n[i].append(history)
				for _ in range(arglist.history_length):
					history_n[i][j].add(obs_n_list[i][j])
		
		# 2.训练
		plt.figure()
		ax = plt.gca()
		print('Starting iterations...')
		while True:
			# 2.1,在num_tasks个任务上进行采样
			for task_index in range(num_tasks):
				action_n = []
				# 用critic获得state,用critic给出action，
				for agent, his in zip(policy, history_n[task_index]):
					# [1, state_dim, length]
					hiss = his.obtain().reshape(1, obs_shape_n[0][0], arglist.history_length)
					action = agent.action([hiss], [1])
					action_n.append(action)
				
				# environment step
				new_obs_n, rew_n, done_n, info_n = list_of_taskenv[task_index].step(action_n)
				local_steps[task_index] += 1		# 更新局部计数器
				global_steps[task_index] += 1		# 更新全局计数器
				
				done = all(done_n)
				terminal = (local_steps[task_index] >= arglist.max_episode_len)
				# 收集experience
				for i, agent in enumerate(model_list[task_index]):
					agent.experience(obs_n_list[task_index][i], action_n[i], rew_n[i], done_n[i], terminal)
				# 更新obs
				obs_n_list[task_index] = new_obs_n

				# 用于test
				if arglist.display:
					time.sleep(0.1)
					list_of_taskenv[task_index].render()
					continue
				
				# 2.2，优化每一个任务的critic
				for i, rew in enumerate(rew_n):
					episodes_rewards[-1] += rew
					agent_rewards[i][-1] += rew
				
				for critic in model_list[task_index]:
					critic.preupdate()
				for critic in model_list[task_index]:
					critic.update(model_list[task_index], global_steps[task_index])
				
				# 2.3，优化actor
				policy_step += 1
				# print("policy steps: ", policy_step)
				for actor, critic in zip(policy, model_list[task_index]):
					actor.add_critic(critic.name)
					actor.update(policy, policy_step)
				
				# 2.4 记录和更新train信息
				# - energy
				energy_one_episode.append(list_of_taskenv[task_index].get_energy())
				# - fair index
				j_index_one_episode.append(list_of_taskenv[task_index].get_jain_index())
				# - coverage
				aver_cover_one_episode.append(list_of_taskenv[task_index].get_aver_cover())
				# - over map counter
				over_map_counter += list_of_taskenv[task_index].get_over_map()
				over_map_one_episode.append(over_map_counter)
				# - disconnected counter
				disconnected_number_counter += list_of_taskenv[task_index].get_dis()
				disconnected_number_one_episode.append(disconnected_number_counter)
				# - reward
				episode_reward_step += np.mean(rew_n)
				accmulated_reward_one_episode.append(episode_reward_step)
				# - state
				s_route = list_of_taskenv[task_index].get_agent_pos()
				for route_i in range(0, FLAGS.num_uav * 2, 2):
					tmp = [s_route[route_i], s_route[route_i + 1]]
					route.append(tmp)
				
				# 当前episode结束是否结束
				if done or terminal:
					# 重置局部变量
					obs_n_list[task_index] = env.reset()		# 重置env
					list_of_taskenv[task_index].set_map(sample_map("../data/chengdu_" + str(i + 1) + ".h5"))
					local_steps[task_index] = 0		# 重置局部计数器
					episodes_rewards.append(0)		# 添加新的元素
					for rew in agent_rewards:
						rew.append(0)
					
					# 记录每个episode的变量
					# - energy
					energy_consumptions_for_test.append(energy_one_episode[-1])
					# - fairness index
					j_index.append(j_index_one_episode[-1])
					# - coverage
					aver_cover.append(aver_cover_one_episode[-1])
					# - disconnected
					instantaneous_dis.append(disconnected_number_one_episode[-1])
					# - out of the map
					instantaneous_out_the_map.append(over_map_one_episode[-1])
					# - reward
					instantaneous_accmulated_reward.append(accmulated_reward_one_episode[-1])
					# - efficiency
					energy_efficiency.append(
						aver_cover_one_episode[-1] * j_index_one_episode[-1] / energy_one_episode[-1])
					print('Task %d, episode: %d - energy_consumptions: %s , energy efficiency : %s.' %
							(
								task_index,
								policy_step / arglist.max_episode_len,
								str(env.get_energy_origin()),
								str(energy_efficiency[-1])
							)
						  )
					
					# 重置每个episode中的局部变量--------------------------------------------
					energy_one_episode = []
					j_index_one_episode = []
					aver_cover_one_episode = []
					over_map_counter = 0
					over_map_one_episode = []
					disconnected_number_counter = 0
					disconnected_number_one_episode = []
					episode_reward_step = 0
					accmulated_reward_one_episode = []
					route = []
				
				# save model, display training output
				episode_number = len(episodes_rewards)
				if terminal and (episode_number % arglist.save_rate == 0):
					save_dir_custom = arglist.save_dir + str(episode_number) + '/'
					U.save_state(save_dir_custom, saver=saver)
					# print statement depends on whether or not there are adversaries
					# 最新save_rate个episode的平均reward
					save_rate_mean_reward = np.mean(episodes_rewards[-arglist.save_rate:])
					if num_adversaries == 0:
						print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
							global_steps[i], episode_number, save_rate_mean_reward,
							round(time.time() - t_start, 3)))
					else:
						print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
							global_steps[i], episode_number, save_rate_mean_reward,
							[np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
					
					t_start = time.time()
					
					final_ep_rewards.append(save_rate_mean_reward)
					for rew in agent_rewards:
						final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
						
					# summary reward
					tf.get_default_session().run(
						final_ep_rewards_sum_assign_op,
						feed_dict={final_ep_rewards_sum_ph: save_rate_mean_reward}
					)
					tf.get_default_session().run(final_ep_rewards_sum_op)
					
					# 绘制reward曲线
					plt.ion()
					ax.plot(np.arange(0, episode_number, arglist.save_rate), final_ep_rewards)
					plt.xlabel("episode number")
					plt.ylabel("reward")
					plt.ioff()
					# 保存train曲线
					if arglist.draw_picture_train:
						model_name = arglist.save_dir.split('/')[-2] + '/'
						draw_util.draw_episode(
							episode_number,
							arglist.pictures_dir_train + model_name,
							aver_cover,
							j_index,
							instantaneous_accmulated_reward,
							instantaneous_dis,
							instantaneous_out_the_map,
							len(aver_cover)
						)

				# saves final episode reward for plotting training curve later
				if episode_number > arglist.num_episodes:
					print('...Finished total of {} episodes.'.format(episode_number))
					break
				

if __name__ == '__main__':
	arglist = parse_args()
	train(arglist)
