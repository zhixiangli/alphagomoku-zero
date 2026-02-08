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
from alphazero.rl import RL
from gomoku.config import GomokuConfig
from gomoku.game import GomokuGame, ChessType
from gomoku.nnet import GomokuNNet


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

    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args
        self.mcts = MCTS(self.nnet, self.game, self.args)

    def next(self, sgf, player):
        actions, counts = self.mcts.simulate(sgf, player)
        pi = counts / sum(counts)
        index = numpy.argmax(0.75 * pi + 0.25 * numpy.random.dirichlet(0.3 * numpy.ones(len(pi))))
        action = actions[index]
        return {'rowIndex': action // self.args.rows, 'columnIndex': action % self.args.rows}


if __name__ == '__main__':

    def init_logging(logpath, is_battle):
        formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(logpath)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        if not is_battle:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)


    parser = argparse.ArgumentParser()

    # Application flags (not part of game/algorithm config)
    parser.add_argument('-is_battle', type=int, default=0)
    parser.add_argument('-logpath', default='./data/gomoku.log')

    # Game-specific config (Gomoku)
    parser.add_argument('-rows', type=int, default=15)
    parser.add_argument('-columns', type=int, default=15)
    parser.add_argument('-n_in_row', type=int, default=5)

    # AlphaZero common config
    parser.add_argument('-save_checkpoint_path', default='./data/model')
    parser.add_argument('-sample_pool_file', default='./data/samples.pkl')
    parser.add_argument('-persist_interval', type=int, default=50)

    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-lr', type=float, default=5e-3)
    parser.add_argument('-l2', type=float, default=1e-4)
    parser.add_argument('-conv_filters', type=int, default=256)
    parser.add_argument('-conv_kernel', default=(3, 3))
    parser.add_argument('-residual_block_num', type=int, default=2)

    parser.add_argument('-simulation_num', type=int, default=500)
    parser.add_argument('-history_num', type=int, default=2)
    parser.add_argument('-c_puct', type=float, default=1)
    parser.add_argument('-max_sample_pool_size', type=int, default=360000)
    parser.add_argument('-temp_step', type=int, default=2)

    cli_args = parser.parse_args()

    init_logging(cli_args.logpath, cli_args.is_battle)

    # Build typed config from CLI arguments
    config = GomokuConfig(
        rows=cli_args.rows,
        columns=cli_args.columns,
        n_in_row=cli_args.n_in_row,
        simulation_num=cli_args.simulation_num,
        c_puct=cli_args.c_puct,
        temp_step=cli_args.temp_step,
        batch_size=cli_args.batch_size,
        epochs=cli_args.epochs,
        max_sample_pool_size=cli_args.max_sample_pool_size,
        persist_interval=cli_args.persist_interval,
        history_num=cli_args.history_num,
        lr=cli_args.lr,
        l2=cli_args.l2,
        conv_filters=cli_args.conv_filters,
        conv_kernel=cli_args.conv_kernel,
        residual_block_num=cli_args.residual_block_num,
        save_checkpoint_path=cli_args.save_checkpoint_path,
        sample_pool_file=cli_args.sample_pool_file,
    )

    logging.info(config)

    game = GomokuGame(config)
    nnet = GomokuNNet(game, config)
    nnet.load_checkpoint(config.save_checkpoint_path)

    if cli_args.is_battle:
        GomokuBattleAgent(nnet, game, config).start()
    else:
        RL(nnet, game, config).start()
