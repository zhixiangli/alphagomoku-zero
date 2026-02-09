#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Gomoku training entry point.

Usage::

    python -m gomoku.trainer
    python -m gomoku.trainer -rows 9 -columns 9 -n_in_row 4
"""

import argparse

from alphazero.module import AlphaZeroModule
from alphazero.trainer import (
    setup_logging,
    add_alphazero_args,
    extract_alphazero_args,
    run_training,
)
from gomoku import configure_module
from gomoku.config import GomokuConfig
from gomoku.game import GomokuGame


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero on Gomoku")

    # Gomoku-specific arguments
    parser.add_argument("-rows", type=int, default=15)
    parser.add_argument("-columns", type=int, default=15)
    parser.add_argument("-n_in_row", type=int, default=5)
    parser.add_argument("-logpath", default="./data/gomoku.log")

    # Common AlphaZero arguments
    add_alphazero_args(parser)

    cli_args = parser.parse_args()
    setup_logging(cli_args.logpath)

    config = GomokuConfig(
        rows=cli_args.rows,
        columns=cli_args.columns,
        n_in_row=cli_args.n_in_row,
        **extract_alphazero_args(cli_args),
    )

    module = AlphaZeroModule()
    configure_module(module)
    run_training(module, GomokuGame, config)


if __name__ == "__main__":
    main()
