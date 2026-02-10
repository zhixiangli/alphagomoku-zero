#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Tests for alphazero/nnet.py - NNet abstract class and AlphaZeroNNet."""

import glob
import os
import tempfile
import unittest

import numpy
import torch
from dotdict import dotdict

from alphazero.nnet import NNet, AlphaZeroNNet, _AlphaZeroModel, _ResidualBlock
from gomoku.game import GomokuGame


def _make_args(**overrides):
    base = {
        "rows": 3,
        "columns": 3,
        "n_in_row": 2,
        "conv_filters": 16,
        "conv_kernel": (3, 3),
        "residual_block_num": 2,
        "lr": 1e-3,
        "l2": 1e-4,
        "batch_size": 4,
        "epochs": 2,
    }
    base.update(overrides)
    return dotdict(base)


class TestNNetAbstract(unittest.TestCase):
    def test_train_raises(self):
        with self.assertRaises(NotImplementedError):
            NNet().train([])

    def test_predict_raises(self):
        with self.assertRaises(NotImplementedError):
            NNet().predict(None)

    def test_save_checkpoint_raises(self):
        with self.assertRaises(NotImplementedError):
            NNet().save_checkpoint("file")

    def test_load_checkpoint_raises(self):
        with self.assertRaises(NotImplementedError):
            NNet().load_checkpoint("file")


class TestResidualBlock(unittest.TestCase):
    def test_output_shape_matches_input(self):
        block = _ResidualBlock(16, (3, 3))
        x = torch.randn(1, 16, 5, 5)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_skip_connection_preserves_gradient(self):
        block = _ResidualBlock(8, (3, 3))
        x = torch.randn(1, 8, 3, 3, requires_grad=True)
        out = block(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestAlphaZeroModel(unittest.TestCase):
    def test_output_shapes(self):
        model = _AlphaZeroModel(
            rows=5, columns=5, conv_filters=16, conv_kernel=(3, 3), residual_block_num=2
        )
        x = torch.randn(2, 2, 5, 5)
        policy, value = model(x)
        self.assertEqual(policy.shape, (2, 25))  # batch=2, action_space=25
        self.assertEqual(value.shape, (2, 1))

    def test_value_head_range(self):
        """Value head uses tanh, so output should be in [-1, 1]."""
        model = _AlphaZeroModel(
            rows=3, columns=3, conv_filters=8, conv_kernel=(3, 3), residual_block_num=1
        )
        x = torch.randn(10, 2, 3, 3)
        _, value = model(x)
        self.assertTrue(torch.all(value >= -1))
        self.assertTrue(torch.all(value <= 1))

    def test_policy_head_is_logits(self):
        """Policy head returns raw logits (no softmax applied)."""
        model = _AlphaZeroModel(
            rows=3, columns=3, conv_filters=8, conv_kernel=(3, 3), residual_block_num=1
        )
        x = torch.randn(1, 2, 3, 3)
        policy, _ = model(x)
        # Logits can be negative or > 1
        # Just verify it's a raw tensor
        self.assertEqual(policy.shape[1], 9)


class TestAlphaZeroNNet(unittest.TestCase):
    def setUp(self):
        self.args = _make_args()
        self.game = GomokuGame(self.args)
        self.nnet = AlphaZeroNNet(self.game, self.args)

    def test_predict_returns_policy_and_value(self):
        board = self.game.get_canonical_form("", "B")
        policy, value = self.nnet.predict(board)
        self.assertEqual(policy.shape, (9,))
        self.assertEqual(value.shape, ())

    def test_predict_policy_sums_to_one(self):
        board = self.game.get_canonical_form("", "B")
        policy, _ = self.nnet.predict(board)
        self.assertAlmostEqual(float(numpy.sum(policy)), 1.0, places=5)

    def test_predict_policy_non_negative(self):
        board = self.game.get_canonical_form("", "B")
        policy, _ = self.nnet.predict(board)
        self.assertTrue(numpy.all(policy >= 0))

    def test_predict_value_in_range(self):
        board = self.game.get_canonical_form("", "B")
        _, value = self.nnet.predict(board)
        self.assertGreaterEqual(float(value), -1.0)
        self.assertLessEqual(float(value), 1.0)

    def test_train_invalidates_frozen_model(self):
        """After training, the frozen model cache should be cleared."""
        board = self.game.get_canonical_form("", "B")
        # Trigger frozen model creation
        self.nnet.predict(board)
        self.assertIsNotNone(self.nnet._frozen_model)

        # Train with minimal data
        rows, cols = self.args.rows, self.args.columns
        action_space = rows * cols
        data = [
            (numpy.zeros((rows, cols, 2)), numpy.ones(action_space) / action_space, 1.0)
            for _ in range(8)
        ]
        self.nnet.train(data)
        self.assertIsNone(self.nnet._frozen_model)

    def test_predict_rebuilds_frozen_model(self):
        """predict() should automatically rebuild frozen model if None."""
        self.assertIsNone(self.nnet._frozen_model)
        board = self.game.get_canonical_form("", "B")
        self.nnet.predict(board)
        self.assertIsNotNone(self.nnet._frozen_model)

    def test_save_and_load_checkpoint(self):
        tmpdir = tempfile.mkdtemp()
        prefix = os.path.join(tmpdir, "test_ckpt")

        self.nnet.save_checkpoint(prefix)
        files = glob.glob(prefix + "*.pt")
        self.assertEqual(len(files), 1)

        # Load should not raise
        self.nnet.load_checkpoint(prefix)

        for f in files:
            os.remove(f)
        os.rmdir(tmpdir)

    def test_load_checkpoint_no_files(self):
        """Loading from nonexistent path should not raise."""
        tmpdir = tempfile.mkdtemp()
        self.nnet.load_checkpoint(os.path.join(tmpdir, "nonexistent"))
        os.rmdir(tmpdir)

    def test_load_checkpoint_clears_frozen_model(self):
        """Loading a checkpoint should clear the frozen model cache."""
        board = self.game.get_canonical_form("", "B")
        self.nnet.predict(board)
        self.assertIsNotNone(self.nnet._frozen_model)

        tmpdir = tempfile.mkdtemp()
        prefix = os.path.join(tmpdir, "test_ckpt")
        self.nnet.save_checkpoint(prefix)
        self.nnet.load_checkpoint(prefix)
        self.assertIsNone(self.nnet._frozen_model)

        for f in glob.glob(prefix + "*.pt"):
            os.remove(f)
        os.rmdir(tmpdir)


if __name__ == "__main__":
    unittest.main()
