"""Shared pytest fixtures for deterministic and reusable test setup."""

from __future__ import annotations

import random
from typing import Callable

import numpy
import pytest
from dotdict import dotdict


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Keep stochastic tests reproducible and order-independent."""
    seed = 12345
    random.seed(seed)
    numpy.random.seed(seed)


@pytest.fixture
def make_args() -> Callable[..., dotdict]:
    """Create minimal dotdict configs with clear defaults.

    Tests can override only what they need.
    """

    def _make_args(**overrides):
        base = {
            "rows": 3,
            "columns": 3,
            "n_in_row": 2,
            "conv_filters": 16,
            "conv_kernel": (3, 3),
            "residual_block_num": 2,
            "save_checkpoint_path": "./tmp",
            "max_sample_pool_size": 10_000,
            "l2": 1e-4,
            "lr": 1e-3,
            "sample_pool_file": "./tmp/sample_pool.pkl",
            "batch_size": 4,
            "epochs": 1,
        }
        base.update(overrides)
        return dotdict(base)

    return _make_args


@pytest.fixture
def build_sgf() -> Callable[[list[tuple[str, int, int]]], str]:
    """Build SGF-like move strings shared by Gomoku/Connect4 tests."""

    def _build(moves: list[tuple[str, int, int]]) -> str:
        return ";".join(f"{player}[{row:x}{col:x}]" for player, row, col in moves)

    return _build
