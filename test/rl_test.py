#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from unittest.mock import patch
from collections import deque

import numpy
from dotdict import dotdict

from alphazero.nnet import AlphaZeroNNet
from alphazero.rl import RL
from gomoku_9_9.game import GomokuGame, ChessType


class TestRL(unittest.TestCase):
    def setUp(self):
        self.args = dotdict(
            {
                "rows": 7,
                "columns": 7,
                "n_in_row": 2,
                "conv_filters": 16,
                "conv_kernel": (3, 3),
                "residual_block_num": 2,
                "save_checkpoint_path": "./tmp",
                "max_sample_pool_size": 10000,
                "l2": 1e-4,
                "lr": 1e-3,
                "sample_pool_file": "./tmp",
            }
        )
        self.game = GomokuGame(self.args)
        self.nnet = AlphaZeroNNet(self.game, self.args)

    def test_augment_board(self):
        board = self.game.get_canonical_form("B[12];W[02]", ChessType.BLACK)
        augmented = self.game.augment_board(board)
        self.assertEqual(len(augmented), 8)
        # Each augmented board should have the same number of non-zero entries
        orig_nonzero = numpy.count_nonzero(board)
        for aug in augmented:
            self.assertEqual(aug.shape, board.shape)
            self.assertEqual(numpy.count_nonzero(aug), orig_nonzero)
        # Verify the identity transform is included (index 6)
        self.assertTrue(numpy.array_equal(augmented[6], board))
        # Verify all 8 are distinct transformations
        for i in range(8):
            for j in range(i + 1, 8):
                self.assertFalse(numpy.array_equal(augmented[i], augmented[j]))

    def test_augment_policy(self):
        pi = numpy.ones((self.args.rows, self.args.columns))
        pi[1][2] = pi[0][2] = 0
        expected = (
            ((2, 5), (2, 6)),
            ((2, 1), (2, 0)),
            ((5, 4), (6, 4)),
            ((5, 2), (6, 2)),
            ((4, 1), (4, 0)),
            ((4, 5), (4, 6)),
            ((1, 2), (0, 2)),
            ((1, 4), (0, 4)),
        )
        for i, p in enumerate(self.game.augment_policy(pi)):
            self.assertEqual(numpy.count_nonzero(p == 0), 2)
            x = p.reshape(self.args.rows, self.args.columns)
            for loc in expected[i]:
                self.assertEqual(x[loc[0]][loc[1]], 0)

    def test_no_reverse_color_method(self):
        """reverse_color is no longer needed with canonical form."""
        self.assertFalse(hasattr(self.game, "reverse_color"))


class _StubGame:
    def __init__(self):
        self.last_action = None

    def get_initial_state(self):
        return "", ChessType.BLACK

    def get_canonical_form(self, board, player):
        return numpy.zeros((1, 2, 2))

    def next_state(self, board, action, player):
        self.last_action = action
        return "B[00]", ChessType.WHITE

    def is_terminal_state(self, board, action, player):
        return ChessType.BLACK

    def compute_reward(self, winner, player):
        return 1 if winner == player else -1


class _StubMCTS:
    def __init__(self, nnet, game, args):
        pass

    def simulate(self, board, player):
        return numpy.array([0, 1]), numpy.array([10, 1])


class TestRLMoveSelection(unittest.TestCase):
    def test_after_temp_step_uses_greedy_mcts_move(self):
        args = dotdict(
            {
                "rows": 1,
                "columns": 2,
                "temp_step": 0,
                "dirichlet_alpha": 0.3,
                "dirichlet_epsilon": 0.25,
                "max_sample_pool_size": 10,
                "sample_pool_file": os.path.join(tempfile.gettempdir(), "rl_stub.pkl"),
            }
        )
        game = _StubGame()
        rl = RL(nnet=None, game=game, args=args)

        with patch("alphazero.rl.MCTS", _StubMCTS), patch(
            "numpy.random.dirichlet", side_effect=AssertionError("dirichlet should not be called")
        ), patch(
            "numpy.random.choice", side_effect=AssertionError("choice should not be called")
        ):
            rl.play_against_itself()

        self.assertEqual(game.last_action, 0)



class TestRLSamplePool(unittest.TestCase):
    def test_persist_and_read_sample_pool(self):
        """persist_sample_pool and read_sample_pool round-trip correctly."""
        tmpdir = tempfile.mkdtemp()
        sample_file = os.path.join(tmpdir, "samples.pkl")
        args = dotdict(
            {
                "rows": 3,
                "columns": 3,
                "n_in_row": 2,
                "conv_filters": 16,
                "conv_kernel": (3, 3),
                "residual_block_num": 2,
                "save_checkpoint_path": "./tmp",
                "max_sample_pool_size": 100,
                "l2": 1e-4,
                "lr": 1e-3,
                "sample_pool_file": sample_file,
            }
        )
        game = GomokuGame(args)
        nnet = AlphaZeroNNet(game, args)
        rl = RL(nnet, game, args)
        # Add some samples
        test_samples = deque([(numpy.zeros((3, 3, 2)), numpy.ones(9) / 9, 1.0)])
        rl.persist_sample_pool(test_samples)
        self.assertTrue(os.path.exists(sample_file))

        # Read back
        loaded = rl.read_sample_pool()
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 1)
        numpy.testing.assert_array_equal(loaded[0][0], numpy.zeros((3, 3, 2)))

        os.remove(sample_file)
        os.rmdir(tmpdir)

    def test_read_sample_pool_missing_file(self):
        """read_sample_pool returns None if file does not exist."""
        args = dotdict(
            {
                "rows": 3,
                "columns": 3,
                "n_in_row": 2,
                "conv_filters": 16,
                "conv_kernel": (3, 3),
                "residual_block_num": 2,
                "save_checkpoint_path": "./tmp",
                "max_sample_pool_size": 100,
                "l2": 1e-4,
                "lr": 1e-3,
                "sample_pool_file": os.path.join(
                    tempfile.gettempdir(), "nonexistent_pool.pkl"
                ),
            }
        )
        game = GomokuGame(args)
        nnet = AlphaZeroNNet(game, args)
        rl = RL(nnet, game, args)
        result = rl.read_sample_pool()
        self.assertIsNone(result)

    def test_sample_pool_maxlen(self):
        """Sample pool should respect max_sample_pool_size."""
        args = dotdict(
            {
                "rows": 3,
                "columns": 3,
                "n_in_row": 2,
                "conv_filters": 16,
                "conv_kernel": (3, 3),
                "residual_block_num": 2,
                "save_checkpoint_path": "./tmp",
                "max_sample_pool_size": 5,
                "l2": 1e-4,
                "lr": 1e-3,
                "sample_pool_file": os.path.join(
                    tempfile.gettempdir(), "nonexistent_pool_2.pkl"
                ),
            }
        )
        game = GomokuGame(args)
        nnet = AlphaZeroNNet(game, args)
        rl = RL(nnet, game, args)
        self.assertEqual(rl.sample_pool.maxlen, 5)
        # Add more than maxlen samples
        for i in range(10):
            rl.sample_pool.append(("board", "policy", i))
        self.assertEqual(len(rl.sample_pool), 5)


if __name__ == "__main__":
    unittest.main()
