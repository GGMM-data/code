import sys
sys.path.append("/home/mxxmhh/mxxhcm/code/")
import argparse
from experimental.LSTM_MADDPG_TF2.multiagent.uav.flag import FLAGS
from experimental.LSTM_MADDPG_TF2.experiments.train import train
import subprocess
import os
import time


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    
    parser.add_argument("--gamma", type=float, default=0.80, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=40, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=160, help="number of units in the mlp")
    parser.add_argument("--buffer-size", type=int, default=100, help="buffer capacity")
    parser.add_argument("--num-task", type=int, default=3, help="number of tasks")
    # rnn 长度
    parser.add_argument('--history_length', type=int, default=4, help="how many history states were used")
    
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which models are saved")
    parser.add_argument("--save-dir", type=str, default="./tmp/",
                        help="directory in which models are saved")
    
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

    # Core training parameters
    parser.add_argument("--save-rate", type=int, default=2,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    # Checkpoint
    parser.add_argument("--exp-name", type=str, default="simple_uav", help="name of the experiment")
    
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


if __name__ == '__main__':
    argslist = parse_args()
    params = ["num_task", "history_length", "max_episode_len", "num_episodes", "batch_size", "gamma", "buffer_size", "num_units"]
    save_path = "policy"
    dict_arg = vars(argslist)
    for param in params:
        save_path = save_path + "_" + param + "_" + str(dict_arg[param])
    save_path = save_path + "_" + str(int(time.time()))
    with open(".info.txt", "a+") as f:
        f.write(save_path+"\n")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train(argslist)

