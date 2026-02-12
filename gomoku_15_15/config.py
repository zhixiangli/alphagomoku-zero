#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from dataclasses import dataclass

from alphazero.config import AlphaZeroConfig


@dataclass
class GomokuConfig(AlphaZeroConfig):
    """Gomoku-specific configuration extending the base AlphaZero config.

    Tuned for 15×15 board, 5-in-a-row Gomoku under limited compute.

    Key design choices vs. the base AlphaZero defaults
    ---------------------------------------------------
    Board & rules
      15×15 is the standard Gomoku board size (225 cells), offering a full
      tactical and strategic experience with complex opening theory.

    MCTS (simulation_num, c_puct, temp_step)
      900 simulations give ~4 visits per legal move in the opening —
      necessary search density for the larger 225-cell action space.
      c_puct 1.5 boosts exploration enough to discover tactical threats the
      early policy head may miss.  temp_step 12 keeps the first ~12 moves
      stochastic for opening diversity, then switches to greedy play.

    Network (conv_filters, residual_block_num)
      128 filters provide enough capacity for 15×15 local and medium-range
      patterns while reducing inference cost.  6 residual blocks add depth for
      multi-step tactical patterns (ladders, forks, double threats) that
      span wider areas on the larger board.

    Training (batch_size, epochs, train_interval, lr)
      512 batch size allows earlier training updates while still giving stable
      gradient estimates.  10 epochs reduce overfitting risk when the replay
      buffer is still small.  Training every 10 games balances feedback
      speed with sufficient data collection.  lr 1e-3 with Adam is
      conservative, reducing the risk of policy collapse.

    Replay buffer (max_sample_pool_size)
      200 000 samples accommodate the longer games and richer positions
      generated on a 15×15 board.
    """

    # Number of consecutive stones needed to win
    n_in_row: int = 5

    # 15×15 board: the standard Gomoku board size, offering full tactical
    # and strategic depth with complex opening theory.
    rows: int = 15
    columns: int = 15

    # -- MCTS --------------------------------------------------------------
    # 450 sims provide a practical compute/strength trade-off on 15×15.
    simulation_num: int = 450

    # Higher exploration constant (1.5 vs 1.0) helps the search discover
    # tactical threats (open-4s, double-3s) that a weak early policy misses.
    c_puct: float = 1.5

    # Stochastic move selection for the first 12 moves (~10–15% of game).
    # Provides diverse openings for training without degrading mid-game play.
    temp_step: int = 12

    # -- Network architecture -----------------------------------------------
    # 128 filters keep inference/training cost manageable while still
    # modeling useful local and medium-range patterns on a 15×15 board.
    conv_filters: int = 128

    # 6 residual blocks give the depth needed for multi-step tactical
    # patterns (ladders, forks, double threats) on a 15×15 board.
    residual_block_num: int = 6

    # -- Training -------------------------------------------------------------
    # 512 batch size enables earlier updates while remaining large enough
    # for stable optimization on augmented self-play samples.
    batch_size: int = 512

    # 10 epochs reduce overfitting risk when the replay buffer is small.
    epochs: int = 10

    # Train every 10 self-play games for balanced feedback loops.
    train_interval: int = 10

    # Conservative learning rate reduces policy collapse risk with Adam.
    lr: float = 1e-3

    # 200 K samples accommodate longer games and richer positions
    # while keeping replay-memory growth bounded.
    max_sample_pool_size: int = 200000

    # Game-specific persistence paths
    save_checkpoint_path: str = "./gomoku_15_15/data/model"
    sample_pool_file: str = "./gomoku_15_15/data/samples.pkl"
