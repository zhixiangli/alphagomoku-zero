#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from dataclasses import dataclass

from alphazero.config import AlphaZeroConfig


@dataclass
class GomokuConfig(AlphaZeroConfig):
    """Gomoku-specific configuration extending the base AlphaZero config.

    Tuned for 9×9 board, 5-in-a-row Gomoku under limited compute.

    Key design choices vs. the base AlphaZero defaults
    ---------------------------------------------------
    Board & rules
      9×9 keeps the action space small (81 cells) while retaining genuine
      Gomoku tactics (open-3s, open-4s, double-threat patterns).

    MCTS (simulation_num, c_puct, temp_step)
      400 simulations keep per-move search robust on a compact 9×9 board
      while maintaining substantially faster self-play iteration than 15×15.
      c_puct 1.5 boosts exploration enough to discover tactical threats the
      early policy head may miss.  temp_step 8 keeps the first ~20 % of moves
      stochastic for opening diversity, then switches to greedy play.

    Network (conv_filters, residual_block_num)
      128 filters (vs 256) halve parameter count and inference time while still
      capturing 9×9 patterns.  4 residual blocks (vs 2) add the depth needed
      for multi-step threat detection without excessive cost.

    Training (batch_size, epochs, train_interval, lr)
      512 batch size lets training start after only 2 self-play games (with 8×
      augmentation).  10 epochs reduce overfitting risk when the replay buffer
      is still small.  Training every 10 games (vs 20) gives faster feedback.
      lr 1e-3 with Adam is more conservative than 5e-3, reducing the risk of
      policy collapse.

    Replay buffer (max_sample_pool_size)
      100 000 samples are sufficient for the 81-cell board and keep memory use
      modest; older data becomes stale quickly as the network improves.
    """

    # Number of consecutive stones needed to win
    n_in_row: int = 5

    # 9×9 board: small enough for fast self-play, large enough for
    # genuine Gomoku tactics (open-3s, double threats, ladder patterns).
    rows: int = 9
    columns: int = 9

    # -- MCTS --------------------------------------------------------------
    # 400 simulations provide solid early-game search density on 9×9
    # while preserving fast self-play throughput.
    simulation_num: int = 400

    # Higher exploration constant (1.5 vs 1.0) helps the search discover
    # tactical threats (open-4s, double-3s) that a weak early policy misses.
    c_puct: float = 1.5

    # Stochastic move selection for the first 8 moves (~20 % of game).
    # Provides diverse openings for training without degrading mid-game play.
    temp_step: int = 8

    # -- Network architecture -----------------------------------------------
    # 128 filters are sufficient for 9×9 patterns and halve the parameter
    # count vs 256, speeding up both training and MCTS inference.
    conv_filters: int = 128

    # 4 residual blocks give the depth needed for multi-step tactical
    # patterns (ladders, forks) without excessive compute on a 9×9 board.
    residual_block_num: int = 4

    # -- Training -------------------------------------------------------------
    # Smaller batches allow earlier training updates while keeping gradient
    # estimates stable with augmented self-play samples.
    batch_size: int = 512

    # 10 epochs reduce overfitting risk when the replay buffer is small.
    epochs: int = 10

    # Train every 10 self-play games for tighter feedback loops.
    train_interval: int = 10

    # Conservative learning rate reduces policy collapse risk with Adam.
    lr: float = 1e-3

    # 100 K samples are enough for a 9×9 board; stale data is evicted
    # quickly so the network always trains on relevant positions.
    max_sample_pool_size: int = 100000

    # Game-specific persistence paths
    save_checkpoint_path: str = "./gomoku_9_9/data/model"
    sample_pool_file: str = "./gomoku_9_9/data/samples.pkl"
