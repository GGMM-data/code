import sys
import os

sys.path.append(os.getcwd() + "/../")

import argparse
from experiments.train import train
from experiments.train_test import test as train_test, multi_process_test as train_multi_process_test
from experiments.transfer_test import test as transfer_test, multi_process_test as transfer_multi_process_test, random_maddpg_test as random_maddpg_test
from multiagent.uav.flag import FLAGS
import os


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # multi thread
    parser.add_argument("--mp", action="store_true", default=False, help="multiprocess test")
    parser.add_argument("--reward-type", type=int, default=2, help="different reward")
    parser.add_argument("--max-test-model-number", type=int, default=900, help="saved max episode number, used for test")
    parser.add_argument("--num-test-episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--save-rate", type=int, default=1,
                        help="save model once every time this many episodes are completed")
    # use lstm
    parser.add_argument('--use-lstm', action="store_true", default=True, help="use lstm?")
    parser.add_argument('--history-length', type=int, default=4, help="how many history states were used")
    # transfer
    parser.add_argument("--num-task", type=int, default=3, help="number of tasks")
    # transfer
    parser.add_argument("--num-task-transfer", type=int, default=1, help="number of tasks")
    # batch size 16
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    # train data name
    parser.add_argument("--train-data-name", type=str, default="chengdu",
                        help="directory in which map data are saved")
    parser.add_argument("--test-data-name", type=str, default="chengdu",
                        help="directory in which map data are saved")
    # not train
    #parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--train", action="store_true", default=False)
    # train test
    #parser.add_argument("--train-test", action="store_true", default=True)
    parser.add_argument("--train-test", action="store_true", default=False)
    # transfer test
    parser.add_argument("--transfer-test", action="store_true", default=True)
    # parser.add_argument("--transfer-test", action="store_true", default=False)

    parser.add_argument("--buffer-size", type=int, default=1000000, help="buffer capacity")
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-train-episodes", type=int, default=4000, help="number of episodes")
    parser.add_argument("--train-data-dir", type=str, default="../../data/train/",
                        help="directory in which map data are saved")
    parser.add_argument("--test-data-dir", type=str, default="../../data/test/",
                        help="directory in which map data are saved")
    
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.83, help="discount factor")
    parser.add_argument("--num-units", type=int, default=160, help="number of units in the mlp")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/",
                        help="directory in which models are saved")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    
    parser.add_argument("--cnn-format", type=str, default='NHWC', help="cnn_format")
    parser.add_argument("--draw-picture-train", action="store_true", default=True)
    parser.add_argument("--draw-picture-test", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="../learning_curves/",
                        help="directory where plot data is saved")
    parser.add_argument("--pictures-dir-train", type=str, default="../result_pictures/train/",
                        help="directory where result pictures data is saved")
    parser.add_argument("--pictures-dir-train-test", type=str, default="../result_pictures/train_test/",
                        help="directory where result pictures data is saved")
    parser.add_argument("--pictures-dir-transfer-test", type=str, default="../result_pictures/transfer_test/",
                        help="directory where result pictures data is saved")

    parser.add_argument("--scenario", type=str, default="simple_uav", help="name of the scenario script")

    #
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="../benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--exp-name", type=str, default="simple_uav", help="name of the experiment")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    return parser.parse_args()


if __name__ == '__main__':
    argslist = parse_args()
    if not argslist.use_lstm:
        argslist.history_length = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    params = ["reward_type", "history_length", "batch_size", "num_task", "train_data_name", "buffer_size",
              "max_episode_len", "save_rate", "gamma", "num_units"]
    save_path = "policy"
    dict_arg = vars(argslist)
    for param in params:
        save_path = save_path + "_" + param + "_" + str(dict_arg[param])
    save_path += "_UAVnumber_" + str(FLAGS.num_uav) + "_size_map_" + str(FLAGS.size_map) + "_radius_" + str(FLAGS.radius)
    argslist.save_dir = argslist.save_dir + save_path + "_debug/"
    argslist.load_dir = argslist.save_dir
    print(argslist.save_dir)
    
    # train
    if argslist.train:
        train(argslist)
        
    # train test
    if argslist.train_test:
        argslist.draw_picture_test = True
        if argslist.mp:
            train_multi_process_test(argslist)
        else:
            train_test(argslist, 300)

    # transfer test
    if argslist.transfer_test:
        argslist.draw_picture_test = True
        if argslist.mp:
            transfer_multi_process_test(argslist)
        else:
            random_maddpg_test(argslist)
            transfer_test(argslist, 300)
