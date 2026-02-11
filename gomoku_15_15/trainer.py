#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Gomoku training entry point.

Usage::

    python -m gomoku_15_15.trainer
    python -m gomoku_15_15.trainer -rows 15 -columns 15 -n_in_row 5
"""

import argparse

from alphazero.module import AlphaZeroModule
from alphazero.trainer import (
    setup_logging,
    add_config_args,
    build_config_from_args,
    run_training,
)
from gomoku_15_15 import configure_module
from gomoku_15_15.config import GomokuConfig
from gomoku_15_15.game import GomokuGame


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero on Gomoku")
    parser.add_argument("-logpath", default="./gomoku_15_15/data/gomoku.log")
    add_config_args(parser, GomokuConfig)

    cli_args = parser.parse_args()
    setup_logging(cli_args.logpath)

    config = build_config_from_args(GomokuConfig, cli_args)

    module = AlphaZeroModule()
    configure_module(module)
    run_training(module, GomokuGame, config)


if __name__ == "__main__":
    main()
