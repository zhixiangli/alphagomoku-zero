#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import argparse
import json
import logging
import sys
from dataclasses import replace
from enum import Enum, unique

import numpy

from alphazero.mcts import MCTS
from alphazero.module import AlphaZeroModule
from alphazero.trainer import setup_logging, add_alphazero_args, extract_alphazero_args, run_training
from gomoku import configure_module
from gomoku.config import GomokuConfig
from gomoku.game import GomokuGame, ChessType


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

    parser = argparse.ArgumentParser()

    # Application flags (not part of game/algorithm config)
    parser.add_argument('-is_battle', type=int, default=0)
    parser.add_argument('-eval', type=int, default=0)
    parser.add_argument('-num_eval_games', type=int, default=50)
    parser.add_argument('-eval_checkpoint_path', default='./data/model2')
    parser.add_argument('-eval_simulation_num', type=int, default=None)
    parser.add_argument('-eval_c_puct', type=float, default=None)
    parser.add_argument('-logpath', default='./data/gomoku.log')

    # Game-specific config (Gomoku)
    parser.add_argument('-rows', type=int, default=15)
    parser.add_argument('-columns', type=int, default=15)
    parser.add_argument('-n_in_row', type=int, default=5)

    # Common AlphaZero arguments
    add_alphazero_args(parser)

    cli_args = parser.parse_args()

    setup_logging(cli_args.logpath, console=not cli_args.is_battle)

    # Build typed config from CLI arguments
    config = GomokuConfig(
        rows=cli_args.rows,
        columns=cli_args.columns,
        n_in_row=cli_args.n_in_row,
        **extract_alphazero_args(cli_args),
    )

    logging.info(config)

    # Wire dependencies through the DI module
    module = AlphaZeroModule()
    configure_module(module)

    if cli_args.is_battle:
        game = GomokuGame(config)
        nnet_class = module.resolve_nnet_class(GomokuGame)
        nnet = nnet_class(game, config)
        nnet.load_checkpoint(config.save_checkpoint_path)
        GomokuBattleAgent(nnet, game, config).start()
    elif cli_args.eval:
        config2 = replace(
            config,
            save_checkpoint_path=cli_args.eval_checkpoint_path,
            simulation_num=cli_args.eval_simulation_num if cli_args.eval_simulation_num is not None else config.simulation_num,
            c_puct=cli_args.eval_c_puct if cli_args.eval_c_puct is not None else config.c_puct,
        )
        evaluator = module.create_evaluator(GomokuGame, config, config2)
        evaluator.nnet1.load_checkpoint(config.save_checkpoint_path)
        evaluator.nnet2.load_checkpoint(config2.save_checkpoint_path)
        results = evaluator.evaluate(num_games=cli_args.num_eval_games)
        logging.info("Final results: %s", results)
    else:
        run_training(module, GomokuGame, config)
