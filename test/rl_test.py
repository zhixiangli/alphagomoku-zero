#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import multiprocessing
import os
import tempfile
import unittest
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy
from dotdict import dotdict

from alphazero.nnet import AlphaZeroNNet
from alphazero.rl import (
    RL,
    _init_self_play_worker,
    _self_play_game,
    _worker_logpath,
    play_one_game,
)
from gomoku.game import GomokuGame, ChessType


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


class TestPlayOneGame(unittest.TestCase):
    """Tests for the standalone play_one_game function."""

    def test_returns_samples(self):
        """play_one_game returns a non-empty list of (board, policy, value) tuples."""
        args = dotdict(
            {
                "rows": 3,
                "columns": 3,
                "n_in_row": 2,
                "conv_filters": 16,
                "conv_kernel": (3, 3),
                "residual_block_num": 2,
                "simulation_num": 10,
                "c_puct": 1.0,
                "temp_step": 2,
                "save_checkpoint_path": "./tmp",
                "max_sample_pool_size": 100,
                "l2": 1e-4,
                "lr": 1e-3,
                "sample_pool_file": os.path.join(
                    tempfile.gettempdir(), "play_one_game_test.pkl"
                ),
            }
        )
        game = GomokuGame(args)
        nnet = AlphaZeroNNet(game, args)
        samples = play_one_game(game, nnet, args)
        self.assertGreater(len(samples), 0)
        for board, policy, value in samples:
            self.assertEqual(board.shape, (3, 3, 2))
            self.assertEqual(policy.shape, (9,))
            self.assertIn(value, [-1, 0, 1])


class TestParallelSelfPlay(unittest.TestCase):
    """Tests for parallel self-play using ProcessPoolExecutor."""

    def test_parallel_games_produce_samples(self):
        """Multiple games run in parallel via ProcessPoolExecutor."""
        args = dotdict(
            {
                "rows": 3,
                "columns": 3,
                "n_in_row": 2,
                "conv_filters": 16,
                "conv_kernel": (3, 3),
                "residual_block_num": 2,
                "simulation_num": 10,
                "c_puct": 1.0,
                "temp_step": 2,
                "save_checkpoint_path": "./tmp",
                "max_sample_pool_size": 100,
                "l2": 1e-4,
                "lr": 1e-3,
                "sample_pool_file": os.path.join(
                    tempfile.gettempdir(), "parallel_test.pkl"
                ),
            }
        )
        game = GomokuGame(args)
        nnet = AlphaZeroNNet(game, args)
        model_state = nnet.model.state_dict()

        num_games = 3
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=mp_context,
            initializer=_init_self_play_worker,
            initargs=(GomokuGame, AlphaZeroNNet, model_state, args),
        ) as executor:
            results = list(executor.map(_self_play_game, range(num_games)))

        self.assertEqual(len(results), num_games)
        for samples in results:
            self.assertGreater(len(samples), 0)
            for board, policy, value in samples:
                self.assertEqual(board.shape, (3, 3, 2))
                self.assertEqual(policy.shape, (9,))


class TestWorkerLogpath(unittest.TestCase):
    """Tests for per-worker log path derivation."""

    def test_worker_logpath_includes_pid(self):
        """_worker_logpath embeds the current PID into the filename."""
        result = _worker_logpath("./data/train.log")
        expected = f"./data/train.worker-{os.getpid()}.log"
        self.assertEqual(result, expected)

    def test_worker_logpath_no_extension(self):
        """_worker_logpath handles paths without an extension."""
        result = _worker_logpath("./data/train")
        expected = f"./data/train.worker-{os.getpid()}"
        self.assertEqual(result, expected)


class TestWorkerLogging(unittest.TestCase):
    """Tests for logging setup inside worker processes."""

    def test_parallel_games_create_worker_logs(self):
        """Worker processes should create per-process log files."""
        tmpdir = tempfile.mkdtemp()
        logpath = os.path.join(tmpdir, "test.log")
        args = dotdict(
            {
                "rows": 3,
                "columns": 3,
                "n_in_row": 2,
                "conv_filters": 16,
                "conv_kernel": (3, 3),
                "residual_block_num": 2,
                "simulation_num": 10,
                "c_puct": 1.0,
                "temp_step": 2,
                "save_checkpoint_path": "./tmp",
                "max_sample_pool_size": 100,
                "l2": 1e-4,
                "lr": 1e-3,
                "sample_pool_file": os.path.join(tmpdir, "samples.pkl"),
                "logpath": logpath,
            }
        )
        game = GomokuGame(args)
        nnet = AlphaZeroNNet(game, args)
        model_state = nnet.model.state_dict()

        num_games = 2
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=mp_context,
            initializer=_init_self_play_worker,
            initargs=(GomokuGame, AlphaZeroNNet, model_state, args),
        ) as executor:
            list(executor.map(_self_play_game, range(num_games)))

        # At least one worker log file should have been created
        worker_logs = [
            f for f in os.listdir(tmpdir) if f.startswith("test.worker-")
        ]
        self.assertGreater(len(worker_logs), 0)
        # Each worker log should contain data (the "winner" log line)
        for wlog in worker_logs:
            wpath = os.path.join(tmpdir, wlog)
            with open(wpath) as f:
                content = f.read()
            self.assertIn("winner", content)

        import shutil
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
