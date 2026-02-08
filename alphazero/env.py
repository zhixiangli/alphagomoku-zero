#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Abstract base class defining the game environment interface for AlphaZero.

This module provides the :class:`Env` interface that must be implemented for
any two-player, zero-sum, perfect-information board game to be used with the
AlphaZero reinforcement-learning framework.  Concrete implementations (e.g.,
:class:`~gomoku.env.GomokuEnv`) define the rules, state transitions, and
termination conditions of a specific game.

The interface is consumed by:

* :class:`~alphazero.mcts.MCTS` – for tree-search rollouts (``next_state``,
  ``is_terminal_state``, ``available_actions``, ``next_player``,
  ``log_status``).
* :class:`~alphazero.rl.RL` – for self-play episode generation
  (``get_initial_state``, ``next_state``, ``is_terminal_state``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Env(ABC):
    """Abstract game environment interface for the AlphaZero framework.

    Every method corresponds to a fundamental operation that the MCTS planner
    and the reinforcement-learning loop need in order to interact with an
    arbitrary two-player board game.

    **Type conventions used throughout the interface:**

    * *Board* – a **hashable** representation of the game state (used as a
      dictionary key inside MCTS).  For example, Gomoku encodes the board as
      a semicolon-delimited SGF string.
    * *Player* – a value that uniquely identifies a player (e.g., ``'B'`` /
      ``'W'``).  The "empty" or "draw" sentinel must be distinct from any
      valid player identifier.
    * *Action* – an ``int`` index that encodes a board position.  Valid
      action values lie in ``[0, rows * columns)``.
    """

    @abstractmethod
    def next_player(self, player: Any) -> Any:
        """Return the opponent of *player*.

        Parameters
        ----------
        player:
            The current player.  Must **not** be the "empty" / draw sentinel.

        Returns
        -------
        Any
            The player whose turn follows *player*.
        """
        ...

    @abstractmethod
    def next_state(
        self, board: Any, action: int, player: Any
    ) -> tuple[Any, Any]:
        """Apply *action* by *player* to *board* and return the successor state.

        Parameters
        ----------
        board:
            The current board state (hashable).
        action:
            The move to apply, encoded as an integer index in
            ``[0, rows * columns)``.
        player:
            The player making the move.

        Returns
        -------
        tuple[Any, Any]
            A ``(new_board, next_player)`` pair where *new_board* is the
            updated board state and *next_player* is the player to move next.
        """
        ...

    @abstractmethod
    def is_terminal_state(
        self, board: Any, action: int, player: Any
    ) -> Any | None:
        """Check whether *board* represents a finished game.

        The caller guarantees that *action* was the last move made by *player*
        that produced *board*.

        Parameters
        ----------
        board:
            The board state to evaluate.
        action:
            The most recent action that led to *board*.
        player:
            The player who performed *action*.

        Returns
        -------
        Any | None
            * The winning player if the game has been won.
            * A draw sentinel (e.g., ``ChessType.EMPTY``) if the game is drawn.
            * ``None`` if the game is still in progress.
        """
        ...

    @abstractmethod
    def get_initial_state(self) -> tuple[Any, Any]:
        """Return the starting state of the game.

        Returns
        -------
        tuple[Any, Any]
            A ``(board, first_player)`` pair representing an empty board and
            the player who moves first.
        """
        ...

    @abstractmethod
    def available_actions(self, board: Any) -> list[int]:
        """Return the legal actions for the given *board*.

        Parameters
        ----------
        board:
            The current board state.

        Returns
        -------
        list[int]
            Legal move indices.  These values are used to index into
            probability / visit-count arrays, so they must be valid indices
            for an array of size ``rows * columns``.
        """
        ...

    @abstractmethod
    def log_status(
        self,
        board: Any,
        counts: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """Log a human-readable snapshot of the board and MCTS statistics.

        Called by :class:`~alphazero.mcts.MCTS` after each simulation round
        for debugging and analysis.

        Parameters
        ----------
        board:
            The current board state.
        counts:
            Visit counts from the MCTS simulation for each available action
            (``float64`` array).
        actions:
            The action indices corresponding to *counts* (``int`` array).
        """
        ...
