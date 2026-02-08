#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from dataclasses import dataclass

from alphazero.config import AlphaZeroConfig


@dataclass
class GomokuConfig(AlphaZeroConfig):
    """Gomoku-specific configuration extending the base AlphaZero config.

    Adds Gomoku-specific parameters like the number of consecutive stones
    needed to win (n_in_row).
    """

    # Number of consecutive stones needed to win
    n_in_row: int = 5

    # Default Gomoku board size
    rows: int = 15
    columns: int = 15
