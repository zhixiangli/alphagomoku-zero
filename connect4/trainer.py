#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Connect4 training entry point.

Usage::

    python -m connect4.trainer
    python -m connect4.trainer -rows 6 -columns 7 -n_in_row 4
"""

import argparse

from alphazero.module import AlphaZeroModule
from alphazero.trainer import (
    setup_logging,
    add_alphazero_args,
    extract_alphazero_args,
    run_training,
)
from connect4 import configure_module
from connect4.config import Connect4Config
from connect4.game import Connect4Game


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero on Connect4")

    # Connect4-specific arguments
    parser.add_argument("-rows", type=int, default=6)
    parser.add_argument("-columns", type=int, default=7)
    parser.add_argument("-n_in_row", type=int, default=4)
    parser.add_argument("-logpath", default="./data/connect4/connect4.log")

    # Common AlphaZero arguments
    add_alphazero_args(
        parser,
        save_checkpoint_path="./data/connect4/model",
        sample_pool_file="./data/connect4/samples.pkl",
    )

    cli_args = parser.parse_args()
    setup_logging(cli_args.logpath)

    config = Connect4Config(
        rows=cli_args.rows,
        columns=cli_args.columns,
        n_in_row=cli_args.n_in_row,
        **extract_alphazero_args(cli_args),
    )

    module = AlphaZeroModule()
    configure_module(module)
    run_training(module, Connect4Game, config)


if __name__ == "__main__":
    main()
