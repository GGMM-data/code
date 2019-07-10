import argparse
import numpy as np
import os
import tensorflow as tf
import time
import pickle
import sys

sys.path.append(os.getcwd() + "/../")

import maddpg_.common.tf_util as U
from experiments.ops import time_begin, time_end, make_env, mlp_model, get_trainers
from experiments.uav_statistics import draw_util
from multiagent.uav.flag import FLAGS


def train(arglist):
    debug = False
    arglist.save_dir = arglist.save_dir + "_batch_size_" + str(arglist.batch_size) + "_buffer_size_" + str(arglist.buffer_size)
    with U.single_threaded_session():
        if debug:
            begin = time_begin()
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        if debug:
            print(time_end(begin, "step 0"))
            begin = time_begin()
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        if debug:
            print(time_end(begin, "step 1"))
            begin = time_begin()
        trainers = get_trainers(env, "task_", num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        if debug:
            print(time_end(begin, "step2"))
            begin = time_begin()
        
        efficiency = tf.placeholder(tf.float32, shape=None, name="efficiency_placeholder")
        efficiency_summary = tf.summary.scalar("efficiency", efficiency)
        p_losses_ph = tf.placeholder(tf.float32, shape=[env.n], name="p_loss")
        p_losses_summary = tf.summary.histogram("loss", p_losses_ph)
        q_losses_ph = tf.placeholder(tf.float32, shape=[env.n], name="q_loss")
        q_losses_summary = tf.summary.histogram("loss", q_losses_ph)
        loss_summary = tf.summary.merge([q_losses_summary, p_losses_summary], name="loss")
        writer = tf.summary.FileWriter("../summary/efficiency")
        writer2 = tf.summary.FileWriter("../summary/loss")

        # Initialize
        U.initialize()
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var)
        if debug:
            print(time_end(begin, "step3"))
            begin = time_begin()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)
        if not os.path.exists(arglist.save_dir):
            os.makedirs(arglist.save_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        model_number = int(arglist.num_episodes / arglist.save_rate)
        saver = tf.train.Saver(max_to_keep=model_number)
        episode_step = 0
        train_step = 0
        t_start = time.time()
        # custom statistics variable------------------------------------------------------------------------------------
        loss_all = []
        aver_cover = []
        j_index = []
        instantaneous_accmulated_reward = []
        instantaneous_dis = []
        instantaneous_out_the_map = []
        # q_value = []
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
        obs_n = env.reset()
        
        episode_reward_step = 0

        model_name = arglist.load_dir.split('/')[-3] + '/' + arglist.load_dir.split('/')[-2] + '/'
        if FLAGS.greedy_action:
            model_name = model_name + 'greedy/'
        elif FLAGS.random_action:
            model_name = model_name + 'random/'
        
        # if debug:
        #     print(time_end(begin, "initialize"))
        #     begin = time_begin()
        print('Starting iterations...')
        episode_begin_time = time.time()
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # increment custom statistics variables in the epoch--------------------------------------------------------
            episode_reward_step += np.mean(rew_n)
            j_index_one_episode.append(env.get_jain_index())
            over_map_counter += env.get_over_map()
            over_map_one_episode.append(over_map_counter)
            disconnected_number_counter += env.get_dis()
            disconnected_number_one_episode.append(disconnected_number_counter)
            aver_cover_one_episode.append(env.get_aver_cover())
            energy_one_episode.append(env.get_energy())
            s_route = env.get_state()
            for route_i in range(0, FLAGS.num_uav * 2, 2):
                tmp = [s_route[route_i], s_route[route_i + 1]]
                route.append(tmp)
            accmulated_reward_one_episode.append(episode_reward_step)
            # if debug:
            #     print(time_end(begin, "others"))
            #     begin = time_begin()
            if done or terminal:
                model_name = arglist.save_dir.split('/')[-1] + '/'
                episode_number = int(train_step / arglist.max_episode_len)
                temp_efficiency = np.array(aver_cover_one_episode) * np.array(
                    j_index_one_episode) / np.array(energy_one_episode)
                draw_util.draw_single_episode(arglist.pictures_dir_train + model_name + "single_episode/",
                                              episode_number,
                                              temp_efficiency,
                                              aver_cover_one_episode,
                                              j_index_one_episode,
                                              energy_one_episode,
                                              disconnected_number_one_episode,
                                              over_map_one_episode,
                                              accmulated_reward_one_episode)

                # reset custom statistics variabl between episode and epoch---------------------------------------------
                instantaneous_accmulated_reward.append(accmulated_reward_one_episode[-1])
                j_index.append(j_index_one_episode[-1])
                instantaneous_dis.append(disconnected_number_one_episode[-1])
                instantaneous_out_the_map.append(over_map_one_episode[-1])
                aver_cover.append(aver_cover_one_episode[-1])
                energy_consumptions_for_test.append(energy_one_episode[-1])
                energy_efficiency.append(aver_cover_one_episode[-1] * j_index_one_episode[-1] / energy_one_episode[-1])
                episode_end_time = time.time()
                
                # plot fig
                efficiency_s = tf.get_default_session().run(efficiency_summary, feed_dict={efficiency: energy_efficiency[episode_number]})
                writer.add_summary(efficiency_s, global_step=episode_number)
                # plt fig
                print('Episode: %d - energy_consumptions: %s, efficiency: %s, time %s' % (train_step / arglist.max_episode_len,
                                                            str(env.get_energy_origin()),
                                                            str(energy_efficiency[-1]),
                                                            str(round(episode_end_time - episode_begin_time, 3))))
                episode_begin_time = episode_end_time
                # draw picture of this episode
                if arglist.draw_picture_test and aver_cover[-1] >= bl_coverage and j_index[-1] >= bl_jainindex \
                        and instantaneous_dis[-1] <= bl_loss:
                    episode_number_name = 'episode_' + str(episode_number)
                    draw_util.draw(episode_number_name, arglist.pictures_dir_test + model_name, energy_one_episode,
                                   route, actions, aver_cover_one_episode, j_index_one_episode,
                                   accmulated_reward_one_episode, disconnected_number_one_episode, over_map_one_episode,
                                   arglist.max_episode_len)

                j_index_one_episode = []
                over_map_counter = 0
                over_map_one_episode = []
                disconnected_number_counter = 0
                disconnected_number_one_episode = []
                aver_cover_one_episode = []
                energy_one_episode = []
                route = []
                episode_reward_step = 0
                accmulated_reward_one_episode = []

                if arglist.draw_picture_test:
                    if len(episode_rewards) % arglist.save_rate == 0:
                        episode_number_name = train_step / arglist.max_episode_len
                        draw_util.drawTest(episode_number_name,
                                           arglist.pictures_dir_train + model_name,
                                           energy_consumptions_for_test,
                                           aver_cover, j_index,
                                           instantaneous_accmulated_reward, instantaneous_dis, instantaneous_out_the_map
                                           , len(aver_cover), bl_coverage, bl_jainindex, bl_loss, energy_efficiency, False)
                # reset custom statistics variabl between episode and epoch---------------------------------------------

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.draw_picture_test:
                if len(episode_rewards) > arglist.num_episodes:
                    break
                continue
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            p_loss_list = []
            q_loss_list = []
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                temp = agent.update(trainers, train_step)
                if temp is not None:
                    p_loss_list.append(temp[1])
                    q_loss_list.append(temp[0])
            if len(p_loss_list) == env.n:
                loss_s = tf.get_default_session().run(loss_summary,
                                                      feed_dict={p_losses_ph: p_loss_list,
                                                                 q_losses_ph: q_loss_list})
                writer2.add_summary(loss_s, global_step=train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                episode_number_name = train_step / arglist.max_episode_len
                save_dir_custom = arglist.save_dir + "/" + str(episode_number_name) + '/'
                # save_dir
                U.save_state(save_dir_custom, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                # draw custom statistics picture when save the model----------------------------------------------------
                if arglist.draw_picture_train:
                    episode_number_name = train_step / arglist.max_episode_len
                    model_name = arglist.save_dir.split('/')[-1] + '/'
                    draw_util.draw_episodes(episode_number_name,
                                            arglist.pictures_dir_train + model_name + "all_episodes/",
                                            aver_cover,
                                            j_index,
                                            energy_consumptions_for_test,
                                            instantaneous_dis,
                                            instantaneous_out_the_map,
                                            energy_efficiency,
                                            instantaneous_accmulated_reward,
                                            len(aver_cover))

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
        # plt.show()

