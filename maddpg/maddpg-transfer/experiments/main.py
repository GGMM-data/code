import sys
import os
cwd = os.getcwd()
path = cwd + "/../"
sys.path.append(path)

import argparse
from experiments.train import train
from multiagent.uav.flag import FLAGS
import os


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    
    parser.add_argument("--gamma", type=float, default=0.83, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=160, help="number of units in the mlp")
    parser.add_argument("--buffer-size", type=int, default=1000000, help="buffer capacity")
    parser.add_argument("--num-task", type=int, default=2, help="number of tasks")
    # rnn 长度
    parser.add_argument('--history_length', type=int, default=1, help="how many history states were used")
    
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which models are saved")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/",
                        help="directory in which models are saved")
    parser.add_argument("--save-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_uav", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=4000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--data-path", type=str, default="../data/chengdu",
                        help="directory in which map data are saved")
    parser.add_argument("--cnn-format", type=str, default='NHWC', help="cnn_format")

    # Core training parameters    # Checkpoint
    parser.add_argument("--exp-name", type=str, default="simple_uav", help="name of the experiment")
    
    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--draw-picture-train", action="store_true", default=True)
    parser.add_argument("--draw-picture-test", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="../learning_curves/",
                        help="directory where plot data is saved")
    parser.add_argument("--pictures-dir-train", type=str, default="../result_pictures/train/",
                        help="directory where result pictures data is saved")
    return parser.parse_args()


if __name__ == '__main__':
    argslist = parse_args()
    params = ["batch_size", "buffer_size", "num_task", "history_length", "max_episode_len", "num_episodes", "save_rate",
              "gamma", "num_units"]
    save_path = "policy"
    dict_arg = vars(argslist)
    for param in params:
        save_path = save_path + "_" + param + "_" + str(dict_arg[param])
    save_path += "_UAVnumber_" + str(FLAGS.num_uav) + "_size_map_" + str(FLAGS.size_map) + "_radius_" + str(FLAGS.radius)
    argslist.save_dir = argslist.save_dir + save_path + "_debug/"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train(argslist)
