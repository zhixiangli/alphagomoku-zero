#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Common training utilities for AlphaZero game training.

Provides reusable helpers for logging setup, CLI argument parsing,
and training orchestration.  Game-specific trainers (e.g.
``gomoku/trainer.py``) add their own arguments, build the game
config, and call :func:`run_training`.
"""

import logging
import os
import sys


def setup_logging(logpath):
    """Configure the root logger with file and console handlers."""
    formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_dir = os.path.dirname(logpath)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(logpath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def add_alphazero_args(parser):
    """Add common AlphaZero config arguments to an argparse parser.

    Game-specific arguments (e.g. board dimensions) should be added
    separately by each game's trainer.
    """
    # Persistence
    parser.add_argument('-save_checkpoint_path', default='./data/model')
    parser.add_argument('-sample_pool_file', default='./data/samples.pkl')
    parser.add_argument('-persist_interval', type=int, default=50)

    # Training
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-lr', type=float, default=5e-3)
    parser.add_argument('-l2', type=float, default=1e-4)

    # Neural network architecture
    parser.add_argument('-conv_filters', type=int, default=256)
    parser.add_argument('-conv_kernel', default=(3, 3))
    parser.add_argument('-residual_block_num', type=int, default=2)

    # MCTS / self-play
    parser.add_argument('-simulation_num', type=int, default=500)
    parser.add_argument('-history_num', type=int, default=2)
    parser.add_argument('-c_puct', type=float, default=1)
    parser.add_argument('-max_sample_pool_size', type=int, default=360000)
    parser.add_argument('-temp_step', type=int, default=2)


def extract_alphazero_args(cli_args):
    """Extract common AlphaZero config fields from parsed CLI args.

    Returns a dict suitable for passing as ``**kwargs`` to any
    :class:`AlphaZeroConfig` subclass constructor.
    """
    return dict(
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


def run_training(module, game_class, config):
    """Create a trainer via the DI module, load checkpoint, and start training.

    This is the standard training entry-point shared by all games.
    """
    logging.info(config)
    trainer = module.create_trainer(game_class, config)
    trainer.nnet.load_checkpoint(config.save_checkpoint_path)
    trainer.start()
