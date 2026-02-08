#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Abstract base class defining the neural network interface for AlphaZero.

This module provides the :class:`NNet` interface that wraps the deep neural
network used by the AlphaZero framework.  Concrete implementations (e.g.,
:class:`~gomoku.nnet.GomokuNNet`) supply the model architecture, training
loop, and weight-persistence logic.

The network jointly predicts two outputs for a given board position:

* **Policy** – a probability distribution over all possible moves, used by
  :class:`~alphazero.mcts.MCTS` as prior probabilities for tree expansion.
* **Value** – a scalar estimate of the expected game outcome, used by MCTS
  to initialise leaf-node values.

The interface is consumed by:

* :class:`~alphazero.mcts.MCTS` – calls :meth:`predict` during leaf
  expansion.
* :class:`~alphazero.rl.RL` – calls :meth:`train` after self-play, and
  :meth:`save_weights` / :meth:`load_weights` for checkpointing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class NNet(ABC):
    """Abstract neural network interface for the AlphaZero framework.

    Subclasses must implement the four methods below so that the MCTS planner
    and the reinforcement-learning loop can train, query, and persist the
    model without depending on a specific deep-learning backend.
    """

    @abstractmethod
    def train(
        self,
        data: Sequence[tuple[Any, Any, np.ndarray, float]],
    ) -> None:
        """Train the network on a batch of self-play examples.

        Parameters
        ----------
        data:
            A sequence of ``(board, player, policy, value)`` tuples where:

            * *board* – the board state representation (hashable).
            * *player* – the player to move.
            * *policy* – a probability distribution over all board positions
              (``float64`` array of length ``rows * columns``) representing
              the MCTS-improved policy target.
            * *value* – the game outcome from the perspective of *player*
              (``+1`` win, ``-1`` loss, ``0`` draw).
        """
        ...

    @abstractmethod
    def predict(self, data: list[Any]) -> tuple[np.ndarray, float]:
        """Predict the policy and value for a single board position.

        Parameters
        ----------
        data:
            A ``[board, player]`` pair describing the position to evaluate.

        Returns
        -------
        tuple[np.ndarray, float]
            A ``(policy, value)`` pair where:

            * *policy* is a probability vector over all board positions
              (``float64`` array of length ``rows * columns``).
            * *value* is a scalar estimate of the expected outcome from the
              perspective of *player* (range ``[-1, 1]``).
        """
        ...

    @abstractmethod
    def save_weights(self, filename: str) -> None:
        """Persist the current network weights to disk.

        Implementations may append a timestamp or epoch number to *filename*
        to keep a history of checkpoints.

        Parameters
        ----------
        filename:
            Base file path for the saved weights.
        """
        ...

    @abstractmethod
    def load_weights(self, filename: str) -> None:
        """Load network weights from disk.

        If no matching weight file is found, the method should return
        silently so that training can start from scratch.

        Parameters
        ----------
        filename:
            Base file path (or glob-pattern prefix) from which to load the
            most recent checkpoint.
        """
        ...
