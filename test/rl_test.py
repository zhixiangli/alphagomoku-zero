#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import unittest

import numpy
from dotdict import dotdict

from gomoku.game import GomokuGame
from gomoku.nnet import GomokuNNet


class TestRL(unittest.TestCase):

    def setUp(self):
        self.args = dotdict({
            'rows': 7,
            'columns': 7,
            'n_in_row': 2,
            'history_num': 2,
            'conv_filters': 16,
            'conv_kernel': (3, 3),
            'residual_block_num': 2,
            'save_checkpoint_path': './tmp',
            'max_sample_pool_size': 10000,
            'l2': 1e-4,
            'lr': 1e-3,
            'sample_pool_file': './tmp',
        })
        self.game = GomokuGame(self.args)
        self.nnet = GomokuNNet(self.game, self.args)

    def test_coordinate_transform(self):
        """
        . . o . o . .
        . . x . x . .
        o x . . . x o
        . . . . . . .
        o x . . . x o
        . . x . x . .
        . . o . o . .
        """
        expected = {(1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)}
        self.assertEqual(expected, set(self.game.augment_coordinate(1, 2)))

    def test_augment_board(self):
        sgf = "B[12];W[02]"
        expected = ['B[25];W[26]', 'B[21];W[20]', 'B[54];W[64]', 'B[52];W[62]', 'B[41];W[40]', 'B[45];W[46]',
                    'B[12];W[02]', 'B[14];W[04]']

        self.assertEqual(expected, self.game.augment_board(sgf))

    def test_augment_policy(self):
        pi = numpy.ones((self.args.rows, self.args.columns))
        pi[1][2] = pi[0][2] = 0
        expected = (
            ((2, 5), (2, 6)), ((2, 1), (2, 0)), ((5, 4), (6, 4)), ((5, 2), (6, 2)), ((4, 1), (4, 0)), ((4, 5), (4, 6)),
            ((1, 2), (0, 2)), ((1, 4), (0, 4)),)
        for i, p in enumerate(self.game.augment_policy(pi)):
            self.assertEqual(numpy.count_nonzero(p == 0), 2)
            x = p.reshape(self.args.rows, self.args.columns)
            for loc in expected[i]:
                self.assertEqual(x[loc[0]][loc[1]], 0)

    def test_no_reverse_color_method(self):
        """reverse_color is no longer needed with canonical form."""
        self.assertFalse(hasattr(self.game, 'reverse_color'))


if __name__ == '__main__':
    unittest.main()
