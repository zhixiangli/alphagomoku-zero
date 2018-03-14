#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import argparse
import json
import logging
import logging.handlers
import sys
from enum import Enum, unique

import numpy

from alphazero.mcts import MCTS
from gomoku.env import GomokuEnv, ChessType
from gomoku.nnet import GomokuNNet
from gomoku.rl import GomokuRL


@unique
class Command(Enum):
    NEXT_BLACK = 1
    NEXT_WHITE = 2


class BattleAgent:
    def start(self):
        while True:
            line = sys.stdin.readline().strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg['command'] == Command.NEXT_BLACK.name:
                print(self.next(msg['chessboard'], ChessType.BLACK), flush=True)
            elif msg['command'] == Command.NEXT_WHITE.name:
                print(self.next(msg['chessboard'], ChessType.WHITE), flush=True)

    def next(self, sgf, player):
        raise NotImplementedError()


class GomokuBattleAgent(BattleAgent):

    def __init__(self, nnet, env, args):
        self.nnet = nnet
        self.env = env
        self.args = args
        self.mcts = MCTS(self.nnet, self.env, self.args)

    def next(self, sgf, player):
        actions, counts = self.mcts.simulate(sgf, player)
        pi = counts / sum(counts)
        index = numpy.argmax(0.75 * pi + 0.25 * numpy.random.dirichlet(0.3 * numpy.ones(len(pi))))
        action = actions[index]
        return {'rowIndex': action // self.args.rows, 'columnIndex': action % self.args.rows}


if __name__ == '__main__':

    def init_logging(args):
        formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(args.logpath)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        if not args.is_battle:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)


    parser = argparse.ArgumentParser()

    parser.add_argument('-rows', type=int, default=15)
    parser.add_argument('-columns', type=int, default=15)
    parser.add_argument('-n_in_row', type=int, default=5)
    parser.add_argument('-is_battle', type=int, default=0)

    parser.add_argument('-save_weights_path', default='./data/model')
    parser.add_argument('-sample_pool_file', default='./data/samples.pkl')
    parser.add_argument('-persist_interval', type=int, default=10)
    parser.add_argument('-logpath', default='./data/gomoku.log')

    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-l2', type=float, default=1e-4)
    parser.add_argument('-conv_filters', type=int, default=64)
    parser.add_argument('-conv_kernel', default=(3, 3))
    parser.add_argument('-residual_block_num', type=int, default=4)

    parser.add_argument('-simulation_num', type=int, default=400)
    parser.add_argument('-history_num', type=int, default=1)
    parser.add_argument('-c_puct', type=float, default=1)
    parser.add_argument('-max_sample_pool_size', type=int, default=200000)
    parser.add_argument('-temp_step', type=int, default=5)

    args = parser.parse_args()

    init_logging(args)
    logging.info(args)

    env = GomokuEnv(args)
    nnet = GomokuNNet(env, args)
    nnet.load_weights(args.save_weights_path)

    if args.is_battle:
        GomokuBattleAgent(nnet, env, args).start()
    else:
        GomokuRL(nnet, env, args).reinforcement_learning()
