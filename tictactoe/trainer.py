#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Tic-Tac-Toe training entry point.

Usage::

    python -m tictactoe.trainer
    python -m tictactoe.trainer -rows 3 -columns 3 -n_in_row 3
"""

import argparse

from alphazero.module import AlphaZeroModule
from alphazero.trainer import setup_logging, add_alphazero_args, extract_alphazero_args, run_training
from tictactoe import configure_module
from tictactoe.config import TicTacToeConfig
from tictactoe.game import TicTacToeGame


def main():
    parser = argparse.ArgumentParser(description='Train AlphaZero on Tic-Tac-Toe')

    # Tic-Tac-Toe-specific arguments
    parser.add_argument('-rows', type=int, default=3)
    parser.add_argument('-columns', type=int, default=3)
    parser.add_argument('-n_in_row', type=int, default=3)
    parser.add_argument('-logpath', default='./data/tictactoe.log')

    # Common AlphaZero arguments
    add_alphazero_args(parser)

    cli_args = parser.parse_args()
    setup_logging(cli_args.logpath)

    config = TicTacToeConfig(
        rows=cli_args.rows,
        columns=cli_args.columns,
        n_in_row=cli_args.n_in_row,
        **extract_alphazero_args(cli_args),
    )

    module = AlphaZeroModule()
    configure_module(module)
    run_training(module, TicTacToeGame, config)


if __name__ == '__main__':
    main()
