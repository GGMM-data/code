import sys
sys.path.append("/home/mxxmhh/mxxhcm/code/")
import argparse
from experimental.LSTM_MADDPG_TF2.multiagent.uav.flag import FLAGS
from experimental.LSTM_MADDPG_TF2.experiments.train import parse_args, train
import subprocess
import os
import time

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

