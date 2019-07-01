import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle
import sys
sys.path.append("/home/mxxmhh/mxxhcm/code/maddpg/maddpg-tmc-optimize")

import maddpg_.common.tf_util as U
from maddpg_.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from uav_statistics import draw_util
from multiagent.uav.flag import FLAGS
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.83, help="discount f  rractor")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=160, help="number of units in the mlp")
    parser.add_argument("--buffer-size", type=int, default=50000, help="buffer capacity")
    parser.add_argument("--save-dir", type=str, default="./tmp/num_uav_"+str(FLAGS.num_uav)
                                                        + "_batch_" + str(256)
                                                        + "_map_size_" + str(FLAGS.size_map)
                                                        + "_radius_"+str(FLAGS.radius)
                                                        + "_max_speed_" + str(FLAGS.max_speed)
                                                        + "_factor_"+str(FLAGS.factor)
                                                        + "_constrain_" + str(FLAGS.constrain),
                        help="directory in which training state and model should be saved")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_uav", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=4000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    #
    parser.add_argument("--load-dir", type=str, default="./tmp/policy_f_1_u_7_r_3_c_5_with_wall/2599/",
                        help="directory in which training state and model are loaded")
    parser.add_argument("--exp-name", type=str, default="simple_uav", help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
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

    # custom parameters for uav
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment_uav import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def time_begin():
	return time.time()


def time_end(begin_time, info):
	print(info)
	return time.time() - begin_time


def train(arglist):
    debug = True
    with U.single_threaded_session():
        if debug:
            begin = time_begin()
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

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
        obs_n = env.reset()
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
            j_index_one_episode.append(env._get_jain_index())
            over_map_counter += env._get_over_map()
            over_map_one_episode.append(over_map_counter)
            disconnected_number_counter += env._get_dis()
            disconnected_number_one_episode.append(disconnected_number_counter)
            aver_cover_one_episode.append(env._get_aver_cover())
            energy_one_episode.append(env._get_energy())
            s_route = env._get_state()
            for route_i in range(0, FLAGS.num_uav * 2, 2):
                tmp = [s_route[route_i], s_route[route_i + 1]]
                route.append(tmp)
            accmulated_reward_one_episode.append(episode_reward_step)
            # if debug:
            #     print(time_end(begin, "others"))
            #     begin = time_begin()
            if done or terminal:
                episode_number = int(train_step / arglist.max_episode_len)
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
                                                            str(env._get_energy_origin()),
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
                        draw_util.drawTest(episode_number_name, arglist.pictures_dir_test + model_name,
                                           energy_consumptions_for_test, aver_cover, j_index,
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
        # plt.show()
        
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    arglist = parse_args()
    train(arglist)
