#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from dataclasses import dataclass

from alphazero.config import AlphaZeroConfig


@dataclass
class TicTacToeConfig(AlphaZeroConfig):
    """Tic-Tac-Toe-specific configuration extending the base AlphaZero config.

    Adds the number of consecutive stones needed to win (n_in_row)
    and defaults to a 3x3 board.
    """

    # Number of consecutive stones needed to win
    n_in_row: int = 3

    # Default Tic-Tac-Toe board size
    rows: int = 3
    columns: int = 3
