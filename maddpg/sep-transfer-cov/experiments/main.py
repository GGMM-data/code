import sys
import os

sys.path.append(os.getcwd() + "/../")

import argparse
from experiments.train import train
from experiments.test import test
from multiagent.uav.flag import FLAGS
import os


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    # parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=True)
    # parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num-task", type=int, default=3, help="number of tasks")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--buffer-size", type=int, default=1000000, help="buffer capacity")
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-train-episodes", type=int, default=4000, help="number of episodes")
    parser.add_argument("--num-test-episodes", type=int, default=50, help="number of episodes")
    parser.add_argument("--save-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--train-data-dir", type=str, default="../data/train/",
                        help="directory in which map data are saved")
    parser.add_argument("--train-data-name", type=str, default="train",
                        help="directory in which map data are saved")
    parser.add_argument("--test-data-dir", type=str, default="../data/test/",
                        help="directory in which map data are saved")
    parser.add_argument("--test-data-name", type=str, default="test",
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
    parser.add_argument("--plots-dir", type=str, default="../learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--pictures-dir-train", type=str, default="../result_pictures/train/", help="directory where result pictures data is saved")
    parser.add_argument("--pictures-dir-test", type=str, default="../result_pictures/test/", help="directory where result pictures data is saved")
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    params = ["batch_size", "buffer_size", "num_task", "train_data_name", "max_episode_len",
              "save_rate", "gamma", "num_units"]
    save_path = "policy"
    dict_arg = vars(argslist)
    for param in params:
        save_path = save_path + "_" + param + "_" + str(dict_arg[param])
    save_path += "_UAVnumber_" + str(FLAGS.num_uav) + "_size_map_" + str(FLAGS.size_map) + "_radius_" + str(FLAGS.radius)
    argslist.save_dir = argslist.save_dir + save_path + "_debug/"
    argslist.load_dir = argslist.save_dir

    # train
    if argslist.train:
        train(argslist)
    # test
    if test:
        argslist.draw_picture_test = True
        test(argslist)
