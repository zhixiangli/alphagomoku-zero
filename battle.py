#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import argparse

from gomoku.env import GomokuEnv
from gomoku.nnet import GomokuNNet
from gomoku.rl import GomokuRL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-rows', default=15)
    parser.add_argument('-columns', default=15)
    parser.add_argument('-n_in_row', default=5)

    parser.add_argument('-simulation_num', default=600)
    parser.add_argument('-history_num', default=1)
    parser.add_argument('-save_weights_path', default='./data/model')
    parser.add_argument('-c_puct', default=8)
    parser.add_argument('-max_sample_pool_size', default=100000)
    parser.add_argument('-batch_size', default=1024)
    parser.add_argument('-epochs', default=5)
    parser.add_argument('-l2', default=1e-6)
    parser.add_argument('-save_weights_interval', default=10)
    parser.add_argument('-conv_filters', default=64)
    parser.add_argument('-conv_kernel', default=(3, 3))
    parser.add_argument('-residual_block_num', default=9)

    args = parser.parse_args()
    print("parsed args", args)

    env = GomokuEnv(args)
    nnet = GomokuNNet(env, args)
    rl = GomokuRL(nnet, env, args)
    rl.reinforcement_learning()
