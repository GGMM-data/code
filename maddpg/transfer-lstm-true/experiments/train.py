import numpy as np
import os
import tensorflow as tf
import time
import pickle
import sys
import math
import queue

sys.path.append(os.getcwd() + "/../")

import maddpg_.common.tf_util as U
from experiments.ops import mkdir, make_env, get_trainers, sample_map
from experiments.uav_statistics import draw_util


def train(arglist):
    debug = False
    multi_process = arglist.mp
    num_tasks = arglist.num_task  # 总共有多少个任务
    list_of_taskenv = []  # env list
    save_path = arglist.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with U.single_threaded_session():
        sess = tf.get_default_session()
        # 1.1创建每个任务的actor trainer和critic trainer
        env = make_env(arglist.scenario, reward_type=arglist.reward_type)
        env.set_map(sample_map(arglist.train_data_dir + arglist.train_data_name + "_1.h5"))
        
        # Create agent trainers
        num_adversaries = min(env.n, arglist.num_adversaries)
        # map for all agents
        common_obs_shape = env.common_observation_space.shape
        # speed size and direction
        sep_obs_shape = env.sep_observation_space.shape

        lstm_scope = "lstm"
        cnn_scope = "cnn"
        actor_scope = "actor_"

        actor_0 = get_trainers(env=env, env_name=actor_scope,
                               num_adversaries=num_adversaries, arglist=arglist,
                               common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape,
                               lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                               agent_type=0, reuse=False, session=sess)

        # 是否共享lstm
        if not arglist.shared_lstm:
            lstm_scope = None
        if not arglist.shared_cnn:
            cnn_scope = None
        
        # 1.2创建每个任务的actor trainer和critic trainer
        critic_list = []  # 所有任务critic的list
        actor_list = []
        for i in range(num_tasks):
            list_of_taskenv.append(make_env(arglist.scenario, reward_type=arglist.reward_type))
            
            critic_trainers = get_trainers(env=list_of_taskenv[i], env_name="task_" + str(i + 1) + "_",
                                           num_adversaries=num_adversaries, arglist=arglist,
                                           common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape,
                                           lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                                           actors=actor_0, agent_type=1, session=sess)
            
            actor_trainers = get_trainers(actor_scope=actor_scope,
                                          env=list_of_taskenv[i], env_name="task_" + str(i + 1) + "_",
                                          num_adversaries=num_adversaries, arglist=arglist,
                                          common_obs_shape=common_obs_shape, sep_obs_shape=sep_obs_shape,
                                          lstm_scope=lstm_scope, cnn_scope=cnn_scope,
                                          agent_type=2, session=sess)
            
            actor_list.append(actor_trainers)
            critic_list.append(critic_trainers)

        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

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

        global_steps_tensor = tf.Variable(tf.zeros(num_tasks), trainable=False)  # global timesteps for each env
        global_steps_ph = tf.placeholder(tf.float32, [num_tasks])
        global_steps_assign_op = tf.assign(global_steps_tensor, global_steps_ph)
        model_number = int(arglist.num_train_episodes / arglist.save_rate)
        saver = tf.train.Saver(max_to_keep=model_number)

        efficiency_list = []
        for i in range(num_tasks):
            efficiency_list.append(tf.placeholder(tf.float32, shape=None, name="efficiency_placeholder" + str(i)))
        efficiency_summary_list = []
        for i in range(num_tasks):
            efficiency_summary_list.append(tf.summary.scalar("efficiency_%s" % i, efficiency_list[i]))
        writer = tf.summary.FileWriter("../summary/efficiency")

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

        # Initialize
        U.initialize()
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var)
        if debug:
            print(time_end(begin, "step3"))
            begin = time_begin()
            
        # 1.4 加载checkpoints
        if arglist.load_dir == "":
            arglist.load_dir = save_path
        if arglist.display or arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)
        global_steps = tf.get_default_session().run(global_steps_tensor)

        # 1.5 初始化ENV
        obs_n_list = []
        for i in range(num_tasks):
            (common_obs, sep_obs_n) = list_of_taskenv[i].reset()
            list_of_taskenv[i].set_map(
                        sample_map(arglist.train_data_dir + arglist.train_data_name + "_" + str(i + 1) + ".h5"))
            obs_n_list.append((common_obs, sep_obs_n))
               
        if debug:
            print(time_end(begin, "initialize"))
            begin = time_begin()
        # 2.训练
        t_start = time.time()
        print('Starting iterations...')
        episode_start_time = time.time()
        # state_dim = common_obs_shape + sep_obs_shape[0]

        history_n = [queue.Queue(arglist.history_length) for _ in range(num_tasks)]
        for i in range(num_tasks):
            for j in range(arglist.history_length):
                history_n[i].put(obs_n_list[i])

        while True:
            for task_index in range(num_tasks):
                # 2.1更新环境，采集样本
                current_env = list_of_taskenv[task_index]
                # get action
                # action_n = [agent.action(obs) for agent, obs in zip(actor_0, obs_n_list[task_index])]
                action_n = []
                common = []
                sep = []
                for com_obs, sep_obs in history_n[task_index].queue:
                    # com_obs, sep_obs = obs
                    common.append(com_obs)
                    sep.append(sep_obs)
                common = np.stack(common)
                sep = np.array(sep)
                                        
                for index, agent in enumerate(actor_0):
                    action_n.append(agent.action(common, sep[:, index, :]))
                                        
                # environment step
                new_obs_n, rew_n, done_n, info_n = current_env.step(action_n)
                current_critics = critic_list[task_index]
                current_actors = actor_list[task_index]
                if debug:
                    print(time_end(begin, "env.step"))
                    begin = time_begin()
                local_steps[task_index] += 1  # 更新局部计数器
                global_steps[task_index] += 1  # 更新全局计数器
                done = all(done_n)
                terminal = (local_steps[task_index] >= arglist.max_episode_len)
                # 收集experience
                # 把n个agents的experience放到一个agent的buffer中
                current_critics[0].experience(obs_n_list[task_index], action_n, rew_n, new_obs_n,
                                                  done_n, terminal)
                # 更新obs
                obs_n_list[task_index] = new_obs_n
                history_n[task_index].get()
                history_n[task_index].put(new_obs_n)
                # 更新reward
                for i, rew in enumerate(rew_n):
                    episodes_rewards[task_index][-1] += rew
                    agent_rewards[task_index][i][-1] += rew
        
                # 2.2，优化每一个任务的critic and acotr
                for critic in current_critics:
                    critic.preupdate()

                for agent_index, critic in enumerate(current_critics):
                    critic.update(current_critics, global_steps[task_index], agent_index)

                for index, actor in enumerate(current_actors):
                    actor.update(current_actors, current_critics, global_steps[task_index], index)

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
                    with open(arglist.pictures_dir_train + model_name + "task_" + str(task_index) + '_train_info' + '.txt', 'a+') as f:
                        info = "Task index: %d, Episode number %d, energy consumption: %s, efficiency: %s, time: %s" % (
                               task_index, episode_number, str(current_env.get_energy_origin()),
                               str(energy_efficiency[task_index][-1]), str(round(episode_time, 3)))
                        f.write(info+"\n")
                    print(info)

                    # 绘制reward曲线
                    efficiency_s = tf.get_default_session().run(efficiency_summary_list[task_index],
                                                                feed_dict={efficiency_list[task_index]:
                                                                               energy_efficiency[task_index][-1]})
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
                        sample_map(arglist.train_data_dir + arglist.train_data_name + "_" + str(task_index + 1) + ".h5"))
                    local_steps[task_index] = 0  # 重置局部计数器
            
                    # 更新全局变量
                    episodes_rewards[task_index].append(0)  # 添加新的元素
                    for reward in agent_rewards[task_index]:
                        reward.append(0)
        
                # save model, display training output
                if terminal and (episode_number % arglist.save_rate == 0):
                    tf.get_default_session().run(global_steps_assign_op, feed_dict={global_steps_ph: global_steps})
                    save_dir_custom = os.path.join(save_path, str(episode_number), 'model.ckpt')
                    U.save_state(save_dir_custom, saver=saver)
                    # print statement depends on whether or not there are adversaries
                    # 最新save_rate个episode的平均reward
                    save_rate_mean_reward = np.mean(episodes_rewards[task_index][-arglist.save_rate:])
                    if num_adversaries == 0:
                        print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                            global_steps[task_index], episode_number, save_rate_mean_reward,
                            round(time.time() - t_start, 3)))
                    else:
                        print(
                            "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                                global_steps[task_index], episode_number, save_rate_mean_reward,
                                [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards[task_index]],
                                round(time.time() - t_start, 3)))
            
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
                if episode_number > arglist.num_train_episodes:
                    mkdir(arglist.plots_dir)
                    rew_file_name = arglist.plots_dir + arglist.exp_name + str(task_index) + '_rewards.pkl'
                    with open(rew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_rewards, fp)
                    agrew_file_name = arglist.plots_dir + arglist.exp_name + str(task_index) + '_agrewards.pkl'
                    with open(agrew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_ag_rewards, fp)
                        print('...Finished total of {} episodes.'.format(episode_number))
            if episode_number > arglist.num_train_episodes:
                break

