#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    """Base configuration for AlphaZero training, shared across all games.

    Game-specific configurations should subclass this and add their own fields.
    The AlphaZero core (MCTS, RL, NNet) depends only on these common fields.
    """

    # Board dimensions
    rows: int
    columns: int

    # MCTS parameters
    simulation_num: int = 500
    c_puct: float = 1.0

    # Self-play parameters
    temp_step: int = 2

    # Training parameters
    batch_size: int = 1024
    epochs: int = 5
    max_sample_pool_size: int = 360000
    persist_interval: int = 50

    # Neural network architecture
    history_num: int = 2
    lr: float = 5e-3
    l2: float = 1e-4
    conv_filters: int = 256
    conv_kernel: tuple = (3, 3)
    residual_block_num: int = 2

    # Persistence paths
    save_weights_path: str = './data/model'
    sample_pool_file: str = './data/samples.pkl'

    @property
    def action_space_size(self) -> int:
        """Size of the action space for grid-based board games."""
        return self.rows * self.columns
