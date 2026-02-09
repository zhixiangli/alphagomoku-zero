#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import unittest

import numpy
from dotdict import dotdict

from gomoku.game import GomokuGame, ChessType
from alphazero.nnet import AlphaZeroNNet


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


if __name__ == "__main__":
    unittest.main()
