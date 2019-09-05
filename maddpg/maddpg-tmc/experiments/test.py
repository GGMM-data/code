import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle
import json

import maddpg_.common.tf_util as U
from maddpg_.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from uav_statistics import draw_util
from multiagent.uav.flag import FLAGS

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")


    ########
    # to change
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.83, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=160, help="number of units in the mlp")
    parser.add_argument("--buffer-size", type=int, default=1000000, help="buffer capacity")
    # parser.add_argument("--load-dir", type=str, default="./tmp/policy_gamma_0.83_batch_1024_neural_160_buffer_1000000_2/",
    #                     help="directory in which training state and model should be saved")

    parser.add_argument("--load-dir", type=str, default="./tmp/num_uav_"+str(FLAGS.num_uav)+"_radius_"+str(FLAGS.radius)
                                                        +"_factor_1.0/",
                        help="directory in which training state and model are loaded")


    #######

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_uav", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    # parser.add_argument("--train-num-episodes", type=int, default=4000, help="number of episodes during training")
    # parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes")
    #########for piao zong#########
    parser.add_argument("--train-num-episodes", type=int, default=1, help="number of episodes during training")
    parser.add_argument("--num-episodes", type=int, default=1, help="number of episodes")
    #########for piao zong#########
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="simple_uav", help="name of the experiment")
    # ./tmp/policy
    parser.add_argument("--save-dir", type=str, default="./tmp/policy_f_1_u_6_r_1.75_c_5/",
                        help="directory in which training state and model should be saved")

    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--draw-picture-train", action="store_true", default=False)
    parser.add_argument("--draw-picture-test", action="store_true", default=True)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
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


def test(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        model_name = arglist.load_dir.split('/')[-2] + '/'
        if FLAGS.greedy_action:
            model_name = model_name + 'greedy/'
        elif FLAGS.random_action:
            model_name = model_name + 'random/'

        model_index_step = 0
        model_number_total = arglist.train_num_episodes / arglist.save_rate
        max_model_index = 0
        max_average_energy_efficiency = 0
        draw_util.mkdir(arglist.pictures_dir_test + model_name)
        while True:
            # Initialize
            U.initialize()
            if model_index_step >= model_number_total:
                with open(arglist.pictures_dir_test + model_name + 'test_report' + '.txt', 'a+') as file:
                    report = '\nModel ' + str(max_model_index) + ' attained max average energy efficiency' + \
                             '\nMax average energy efficiency:' + str(max_average_energy_efficiency)
                    file.write(report)
                break
            else:
                model_index_step += 1
            # Load previous results, if necessary
            if arglist.load_dir == "":
                arglist.load_dir = arglist.save_dir
            if arglist.display or arglist.restore or arglist.benchmark:
                print('Loading previous state...')
                #model_load_dir = arglist.load_dir + str(model_index_step * arglist.save_rate - 1) + '/'
                model_load_dir = arglist.load_dir + str(3299) + '/'
                U.load_state(model_load_dir)

            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
            final_ep_rewards = []  # sum of rewards for training curve
            final_ep_ag_rewards = []  # agent rewards for training curve
            agent_info = [[[]]]  # placeholder for benchmarking info
            saver = tf.train.Saver()
            obs_n = env.reset()
            episode_step = 0
            train_step = 0
            t_start = time.time()
            # custom statistics variable--------------------------------------------------------------------------------
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
            print('Starting iterations...')
            route_dict = {}
            for i in range(FLAGS.num_uav):
                key_temp = "UAV" + str(i+1)
                route_dict[key_temp] = []
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
                for index, route_i in enumerate(range(0, FLAGS.num_uav * 2, 2)):
                    # for piao zong
                    route_dict["UAV" + str(index+1)].append([s_route[route_i], s_route[route_i + 1]])
                    # for piao zong

                accmulated_reward_one_episode.append(episode_reward_step)

                if done or terminal:
                    ### for piaozong
                    uav_poss_file = "~/UAVNumber_"+str(FLAGS.num_uav) + ".json"
                    route_str = json.dumps(route_dict)
                    with open(uav_poss_file, "w+") as f:
                        f.write(route_str)
                    ### for piaozong

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
                            if np.mean(energy_efficiency) > max_average_energy_efficiency:
                                max_model_index = model_index_step * arglist.save_rate - 1
                                max_average_energy_efficiency = np.mean(energy_efficiency)
                            with open(arglist.pictures_dir_test + model_name + 'test_report' + '.txt', 'a+') as file:
                                report = '\nModel-' + str(model_index_step * arglist.save_rate - 1) + \
                                         '-testing ' + str(arglist.num_episodes) + ' episodes\'s result:' + \
                                         '\nAverage average attained coverage: ' + str(np.mean(aver_cover)) + \
                                         '\nAverage Jaint\'s fairness index: ' + str(np.mean(j_index)) + \
                                         '\nAverage normalized average energy consumptions:' + str(np.mean(energy_consumptions_for_test)) + \
                                         '\nAverage energy efficiency:' + str(np.mean(energy_efficiency)) + '\n'
                                file.write(report)
                            draw_util.drawTest(model_index_step * arglist.save_rate - 1, arglist.pictures_dir_test + model_name,
                                               energy_consumptions_for_test, aver_cover, j_index,
                                               instantaneous_accmulated_reward, instantaneous_dis, instantaneous_out_the_map
                                               , len(aver_cover), bl_coverage, bl_jainindex, bl_loss, energy_efficiency, False)
                    # reset custom statistics variabl between episode and epoch-----------------------------------------

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
    # set to use number 1 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    arglist = parse_args()
    test(arglist)