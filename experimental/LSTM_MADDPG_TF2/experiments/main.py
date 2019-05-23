import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys

sys.path.append("/home/mxxmhh/mxxhcm/code/experimental/")

import experimental.LSTM_MADDPG_TF2.model.common.tf_util as U
from experimental.LSTM_MADDPG_TF2.model.trainer.history import History
from experimental.LSTM_MADDPG_TF2.experiments.uav_statistics import draw_util
from experimental.LSTM_MADDPG_TF2.multiagent.uav.flag import FLAGS
from experimental.LSTM_MADDPG_TF2.experiments.ops import make_env, lstm_model, q_model, mlp_model, get_trainers


def parse_args():
	parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
	
	parser.add_argument("--model-dir", type=str, default="./tmp/policy_gamma_0.80_batch_1024_neural_160_batch_75/",
						help="directory in which training state and model should be saved")
	parser.add_argument("--gamma", type=float, default=0.80, help="discount factor")
	parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
	parser.add_argument("--num-units", type=int, default=160, help="number of units in the mlp")
	parser.add_argument("--buffer-size", type=int, default=100, help="buffer capacity")
	
	# --------------------------------------------------------------
	
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
	
	### add by mxx
	parser.add_argument('--history_length', type=int, default=4, help="how many history states were used")
	parser.add_argument("--cnn-format", type=str, default='NHWC', help="cnn_format")
	
	# custom parameters for uav
	return parser.parse_args()


def train(arglist):
	with U.single_threaded_session():
		# 1.初始化
		num_tasks = 4		# 总共有多少个任务
		list_of_taskenv = []		# env list

		# 1.1创建一个actor
		env = make_env(arglist.scenario, arglist, arglist.benchmark)
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
		policy_step = 0
		model_number = int(arglist.num_episodes / arglist.save_rate)
		saver = tf.train.Saver(max_to_keep=model_number)
		
		# 1.5局部变量初始化
		local_steps = np.zeros(num_tasks)  # local timesteps for each env
		
		episode_rewards = [0.0]  # sum of rewards for all agents
		agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
		final_ep_rewards = []  # sum of rewards for training curve
		final_ep_ag_rewards = []  # agent rewards for training curve
		
		t_start = time.time()
		aver_cover = []
		j_index = []
		instantaneous_accmulated_reward = []
		instantaneous_dis = []
		instantaneous_out_the_map = []
		energy_consumptions_for_test = []
		bl_coverage = 0.8
		bl_jainindex = 0.8
		bl_loss = 100
		energy_efficiency = []
		
		over_map_counter = 0
		over_map_one_episode = []
		aver_cover_one_episode = []
		j_index_one_episode = []
		disconnected_number_counter = 0
		disconnected_number_one_episode = []
		accmulated_reward_one_episode = []
		actions = []
		energy_one_episode = []
		route = []
		
		episode_reward_step = 0
		
		# 1.6初始化ENV
		obs_n_list = []
		for i in range(num_tasks):
			obs_n = list_of_taskenv[i].reset()
			obs_n_list.append(obs_n)
		
		# 1.7 生成maddpg 加上rnn之后的输入seq，
		history_n = [[] for _ in range(num_tasks)]
		for i in range(num_tasks):
			for j in range(len(obs_n_list[i])):  # 生成每个智能体长度为history_length的观测
				history = History(arglist, [obs_shape_n[j][0]])
				history_n[i].append(history)
				for _ in range(arglist.history_length):
					history_n[i][j].add(obs_n_list[i][j])
		
		# 训练
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
				local_steps[i] += 1		# 局部计数器
				global_steps[i] += 1		# 全局计数器
				
				done = all(done_n)
				terminal = (local_steps[task_index] >= arglist.max_episode_len)
				# 收集experience
				for i, agent in enumerate(model_list[task_index]):
					agent.experience(obs_n_list[task_index][i], action_n[i], rew_n[i], done_n[i], terminal)
				obs_n_list[task_index] = new_obs_n
				
				# 第2步，优化每一个任务的critic
				for i, rew in enumerate(rew_n):
					episode_rewards[-1] += rew
					agent_rewards[i][-1] += rew
				
				# 记录train信息
				# fair index
				j_index_one_episode.append(env.get_jain_index())
				# over map counter
				over_map_counter += env.get_over_map()
				over_map_one_episode.append(over_map_counter)
				# disconnected counter
				disconnected_number_counter += env.get_dis()
				disconnected_number_one_episode.append(disconnected_number_counter)
				# coverage
				aver_cover_one_episode.append(env.get_aver_cover())
				# energy
				energy_one_episode.append(env.get_energy())
				# reward
				episode_reward_step += np.mean(rew_n)
				accmulated_reward_one_episode.append(episode_reward_step)
				# state
				s_route = env.get_state()
				for route_i in range(0, FLAGS.num_uav * 2, 2):
					tmp = [s_route[route_i], s_route[route_i + 1]]
					route.append(tmp)
				
				if done or terminal:
					obs_n_list[task_index] = env.reset()
					local_steps[task_index] = 0
					episode_rewards.append(0)
					
					# reset custom statistics variabl between episode and epoch---------------------------------------------
					instantaneous_accmulated_reward.append(accmulated_reward_one_episode[-1])
					j_index.append(j_index_one_episode[-1])
					instantaneous_dis.append(disconnected_number_one_episode[-1])
					instantaneous_out_the_map.append(over_map_one_episode[-1])
					aver_cover.append(aver_cover_one_episode[-1])
					energy_consumptions_for_test.append(energy_one_episode[-1])
					energy_efficiency.append(
						aver_cover_one_episode[-1] * j_index_one_episode[-1] / energy_one_episode[-1])
					print('Episode: %d - energy_consumptions: %s ' % (policy_step / arglist.max_episode_len,
																	  str(env._get_energy_origin())))
					
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
					
					# reset custom statistics variabl between episode and epoch---------------------------------------------
					for i in range(num_tasks):
						obs_n_list[i] = list_of_taskenv[i].reset()
					local_steps[i] += 1
					episode_rewards.append(0)
					for a in agent_rewards:
						a.append(0)
				
				for critic in model_list[task_index]:
					critic.preupdate()
				for critic in model_list[task_index]:
					critic.update(model_list[task_index], global_steps[i])
				
				# 第3步，优化actor
				policy_step += 1
				print("policy steps: ", policy_step)
				for actor, critic in zip(policy, model_list[task_index]):
					actor.add_critic(critic.name)
					actor.update(policy, policy_step)
				
				# save model, display training output
				if terminal and (len(episode_rewards) % arglist.save_rate == 0):
					episode_number_name = global_steps[i] / arglist.max_episode_len
					save_dir_custom = arglist.save_dir + str(episode_number_name) + '/'
					U.save_state(save_dir_custom, saver=saver)
					# print statement depends on whether or not there are adversaries
					if num_adversaries == 0:
						print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
							global_steps[i], len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
							round(time.time() - t_start, 3)))
					else:
						print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
							global_steps[i], len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
							[np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
					t_start = time.time()
					# Keep track of final episode reward
					final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
					for rew in agent_rewards:
						final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
				
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
					

if __name__ == '__main__':
	arglist = parse_args()
	train(arglist)
