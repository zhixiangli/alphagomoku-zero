#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from dataclasses import dataclass

from alphazero.config import AlphaZeroConfig


@dataclass
class Connect4Config(AlphaZeroConfig):
    n_in_row: int = 4

    rows: int = 6
    columns: int = 7

    simulation_num: int = 200
    c_puct: float = 1.5
    temp_step: int = 8

    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    conv_filters: int = 64
    residual_block_num: int = 4

    batch_size: int = 256
    epochs: int = 5
    train_interval: int = 10
    lr: float = 1e-3
    max_sample_pool_size: int = 50000

    save_checkpoint_path: str = "./connect4/data/model"
    sample_pool_file: str = "./connect4/data/samples.pkl"
