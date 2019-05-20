import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys

sys.path.append("/home/mxxmhh/mxxhcm/code/experimental/LSTM_MADDPG_TF2")

import LSTM_MADDPG_TF2.model.common.tf_util as U

from LSTM_MADDPG_TF2.model.trainer.history import History
from LSTM_MADDPG_TF2.experiments.uav_statistics import draw_util
from LSTM_MADDPG_TF2.multiagent.uav.flag import FLAGS
from experiments.ops import make_env, lstm_model, q_model, mlp_model, get_trainers


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
		# 总共有多少个任务
		num_tasks = 4
		list_of_taskenv = []
		
		# 所有任务模型的list
		models = []
		for i in range(num_tasks):
			# 创建每个任务的env
			env = make_env(arglist.scenario, arglist, arglist.benchmark)
			# 从env中获得每个agent的observation space
			obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
			num_adversaries = min(env.n, arglist.num_adversaries)
			# 创建每个任务的智能体
			trainers = get_trainers(env, "task_"+str(i)+"_", num_adversaries, obs_shape_n, arglist)
			models.append(trainers)
			list_of_taskenv.append(env)
		policy = get_trainers(env, "pi_0_", num_adversaries, obs_shape_n, arglist)
		print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
		
		U.initialize()
		
		# 多个任务智能体的obs
		obs_n_list = []
		for i in range(num_tasks):
			obs_n = list_of_taskenv[i].reset()
			obs_n_list.append(obs_n)
		
		history_n = [[] for _ in range(num_tasks)]
		for i in range(num_tasks):
			for j in range(len(obs_n_list[i])):  # 生成每个智能体长度为history_length的观测
				history = History(arglist, obs_shape_n[0])
				history_n[i].append(history)
				for _ in range(arglist.history_length):
					history_n[i][j].add(obs_n_list[i])
				
		print('Starting iterations...')
		while True:
			# n个task
			for i in range(num_tasks):
				# get action
				action_n = []  # 获得n个智能体的动作
				for agent, his in zip(trainers, history_n):
					hiss = his.get().reshape(1, obs_shape_n[0][0], arglist.history_length)
					action = agent.action([hiss], [1])  # 这里打印的class tuple, class list
					action_n.append(action)
				
				# environment step
				new_obs_n, rew_n, done_n, info_n = env.step(action_n)
				episode_step += 1
				done = all(done_n)
				terminal = (episode_step >= arglist.max_episode_len)
				# collect experience
				for i, agent in enumerate(trainers):
					agent.experience(obs_n[i], action_n[i], rew_n[i], done_n[i], terminal)
				obs_n = new_obs_n
				
				for i, rew in enumerate(rew_n):
					episode_rewards[-1] += rew
					agent_rewards[i][-1] += rew
				
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
					obs_n = env.reset()
					episode_step = 0
					episode_rewards.append(0)
					for a in agent_rewards:
						a.append(0)
					agent_info.append([[]])
				
				# increment global step counter
				train_step += 1
				
				# for displaying learned policies
				if arglist.draw_picture_test:
					if len(episode_rewards) > arglist.num_episodes:
						break
					continue
				
				# update all trainers, if not in display or benchmark mode
				loss = None
				for agent in trainers:
					agent.preupdate()
				for agent in trainers:
					loss = agent.update(trainers, train_step)
				
				# save model, display training output
				if terminal and (len(episode_rewards) % arglist.save_rate == 0):
					episode_number_name = train_step / arglist.max_episode_len
					save_dir_custom = arglist.save_dir + str(episode_number_name) + '/'
					U.save_state(save_dir_custom, saver=saver)
					# print statement depends on whether or not there are adversaries
					if num_adversaries == 0:
						print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
							train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
							round(time.time() - t_start, 3)))
					else:
						print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
							train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
							[np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
					t_start = time.time()
					# Keep track of final episode reward
					final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
					for rew in agent_rewards:
						final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
					# draw custom statistics picture when save the model----------------------------------------------------
					if arglist.draw_picture_train:
						episode_number_name = train_step / arglist.max_episode_len
						model_name = arglist.save_dir.split('/')[-2] + '/'
						draw_util.draw_episode(episode_number_name, arglist.pictures_dir_train + model_name, aver_cover,
											   j_index, instantaneous_accmulated_reward, instantaneous_dis,
											   instantaneous_out_the_map, loss_all, len(aver_cover))
				
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
