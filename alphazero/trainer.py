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
from dataclasses import MISSING, fields

from alphazero.config import AlphaZeroConfig

# Build a mapping of field name -> default value from the config dataclass
_DEFAULTS = {f.name: f.default for f in fields(AlphaZeroConfig) if f.default is not MISSING}


def setup_logging(logpath, console=False):
    """Configure the root logger with file and optional console handlers."""
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_dir = os.path.dirname(logpath)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(logpath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def add_alphazero_args(
    parser,
    save_checkpoint_path=_DEFAULTS["save_checkpoint_path"],
    sample_pool_file=_DEFAULTS["sample_pool_file"],
):
    """Add common AlphaZero config arguments to an argparse parser.

    Game-specific arguments (e.g. board dimensions) should be added
    separately by each game's trainer.  Pass *save_checkpoint_path* and
    *sample_pool_file* to override the default persistence paths for a
    particular game.
    """
    # Persistence
    parser.add_argument("-save_checkpoint_path", default=save_checkpoint_path)
    parser.add_argument("-sample_pool_file", default=sample_pool_file)
    parser.add_argument("-persist_interval", type=int, default=_DEFAULTS["persist_interval"])
    parser.add_argument("-train_interval", type=int, default=_DEFAULTS["train_interval"])

    # Training
    parser.add_argument("-batch_size", type=int, default=_DEFAULTS["batch_size"])
    parser.add_argument("-epochs", type=int, default=_DEFAULTS["epochs"])
    parser.add_argument("-lr", type=float, default=_DEFAULTS["lr"])
    parser.add_argument("-l2", type=float, default=_DEFAULTS["l2"])

    # Neural network architecture
    parser.add_argument("-conv_filters", type=int, default=_DEFAULTS["conv_filters"])
    parser.add_argument("-conv_kernel", default=_DEFAULTS["conv_kernel"])
    parser.add_argument("-residual_block_num", type=int, default=_DEFAULTS["residual_block_num"])

    # MCTS / self-play
    parser.add_argument("-simulation_num", type=int, default=_DEFAULTS["simulation_num"])
    parser.add_argument("-c_puct", type=float, default=_DEFAULTS["c_puct"])
    parser.add_argument("-max_sample_pool_size", type=int, default=_DEFAULTS["max_sample_pool_size"])
    parser.add_argument("-temp_step", type=int, default=_DEFAULTS["temp_step"])


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
        train_interval=cli_args.train_interval,
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
