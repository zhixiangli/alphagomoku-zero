#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import argparse

from alphazero.module import AlphaZeroModule
from alphazero.trainer import (
    setup_logging,
    add_config_args,
    build_config_from_args,
    run_training,
)
from connect4 import configure_module
from connect4.config import Connect4Config
from connect4.game import Connect4Game


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero on Connect4")
    parser.add_argument("-logpath", default="./connect4/data/connect4.log")
    add_config_args(parser, Connect4Config)

    cli_args = parser.parse_args()
    setup_logging(cli_args.logpath)

    config = build_config_from_args(Connect4Config, cli_args)

    module = AlphaZeroModule()
    configure_module(module)
    run_training(module, Connect4Game, config)


if __name__ == "__main__":
    main()
