#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from dataclasses import dataclass

from alphazero.config import AlphaZeroConfig


@dataclass
class Connect4Config(AlphaZeroConfig):
    """Connect4-specific configuration extending the base AlphaZero config.

    Adds Connect4-specific parameters like the number of consecutive pieces
    needed to win (n_in_row).
    """

    # Number of consecutive pieces needed to win
    n_in_row: int = 4

    # Default Connect4 board size
    rows: int = 6
    columns: int = 7

    # Game-specific persistence paths
    save_checkpoint_path: str = "./data/connect4/model"
    sample_pool_file: str = "./data/connect4/samples.pkl"
