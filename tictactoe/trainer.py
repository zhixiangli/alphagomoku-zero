#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import argparse
import logging
import sys

from alphazero.module import AlphaZeroModule
from tictactoe import configure_module
from tictactoe.config import TicTacToeConfig
from tictactoe.game import TicTacToeGame


def init_logging(logpath):
    formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(logpath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Application flags
    parser.add_argument('-logpath', default='./data/tictactoe.log')

    # Game-specific config (Tic-Tac-Toe)
    parser.add_argument('-rows', type=int, default=3)
    parser.add_argument('-columns', type=int, default=3)
    parser.add_argument('-n_in_row', type=int, default=3)

    # AlphaZero common config
    parser.add_argument('-save_checkpoint_path', default='./data/tictactoe_model')
    parser.add_argument('-sample_pool_file', default='./data/tictactoe_samples.pkl')
    parser.add_argument('-persist_interval', type=int, default=50)

    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-lr', type=float, default=5e-3)
    parser.add_argument('-l2', type=float, default=1e-4)
    parser.add_argument('-conv_filters', type=int, default=64)
    parser.add_argument('-conv_kernel', default=(3, 3))
    parser.add_argument('-residual_block_num', type=int, default=2)

    parser.add_argument('-simulation_num', type=int, default=200)
    parser.add_argument('-history_num', type=int, default=2)
    parser.add_argument('-c_puct', type=float, default=1)
    parser.add_argument('-max_sample_pool_size', type=int, default=50000)
    parser.add_argument('-temp_step', type=int, default=2)

    cli_args = parser.parse_args()

    init_logging(cli_args.logpath)

    # Build typed config from CLI arguments
    config = TicTacToeConfig(
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

    # Wire dependencies through the DI module
    module = AlphaZeroModule()
    configure_module(module)

    trainer = module.create_trainer(TicTacToeGame, config)
    trainer.nnet.load_checkpoint(config.save_checkpoint_path)
    trainer.start()
