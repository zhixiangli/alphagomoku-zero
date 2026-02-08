#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import unittest

import numpy
from dotdict import dotdict

from gomoku.game import GomokuGame, ChessType
from gomoku.nnet import GomokuNNet


class TestGomoku(unittest.TestCase):

    def setUp(self):
        self.args = dotdict({
            'rows': 3,
            'columns': 3,
            'n_in_row': 2,
            'history_num': 2,
            'conv_filters': 16,
            'conv_kernel': (3, 3),
            'residual_block_num': 2,
            'save_weights_path': './tmp',
            'max_sample_pool_size': 10000,
            'l2': 1e-4,
            'lr': 1e-3,
            'sample_pool_file': './tmp',
        })
        self.game = GomokuGame(self.args)
        self.nnet = GomokuNNet(self.game, self.args)

    def test_next_player(self):
        self.assertEqual(self.game.next_player(ChessType.BLACK), ChessType.WHITE)
        self.assertEqual(self.game.next_player(ChessType.WHITE), ChessType.BLACK)

    def test_next_state(self):
        board, player = self.game.get_initial_state()

        next_board, next_player = self.game.next_state(board, 6, player)
        self.assertEqual(next_player, ChessType.WHITE)
        self.assertEqual(next_board, 'B[20]')

        next_board, next_player = self.game.next_state(next_board, 0, next_player)
        self.assertEqual(next_player, ChessType.BLACK)
        self.assertEqual(next_board, 'B[20];W[00]')

    def test_is_terminal_state(self):
        sgf = ';'.join([ChessType.BLACK + self.game.hex_action(i) for i in range(self.args.rows * self.args.columns)])
        self.assertEqual(self.game.is_terminal_state(sgf, 0, ChessType.BLACK), ChessType.BLACK)

        sgf = ';'.join([ChessType.WHITE + self.game.hex_action(i) for i in range(self.args.rows * self.args.columns)])
        self.assertEqual(self.game.is_terminal_state(sgf, 0, ChessType.BLACK), ChessType.EMPTY)

        boards = ["B[11];B[12]", "B[11];B[21]", "B[11];B[22]", "B[11];B[00]"]
        for sgf in boards:
            self.assertEqual(self.game.is_terminal_state(sgf, 1 * self.args.columns + 1, ChessType.BLACK),
                             ChessType.BLACK)

        sgf = "B[03];B[10]"
        self.assertEqual(self.game.is_terminal_state(sgf, 3, ChessType.BLACK), None)

    def test_available_actions(self):
        sgf = ';'.join([ChessType.BLACK + self.game.hex_action(i) for i in range(self.args.rows * self.args.columns)])
        self.assertEqual(self.game.available_actions(sgf), [])

        sgf = ';'.join(
            [ChessType.BLACK + self.game.hex_action(i) for i in range(0, self.args.rows * self.args.columns, 2)])
        self.assertEqual(self.game.available_actions(sgf), [i for i in range(1, self.args.rows * self.args.columns, 2)])

    def test_player_insensitive_board(self):
        self.assertEqual(self.nnet.player_insensitive_board('B[20];W[00]', ChessType.BLACK), 'B[20];W[00]')
        self.assertEqual(self.nnet.player_insensitive_board('B[20];W[00]', ChessType.WHITE), 'W[20];B[00]')

    def test_fit_transform0(self):
        self.assertTrue(numpy.array_equal(self.nnet.fit_transform('B[20];W[21];B[11]', ChessType.WHITE),
                                          numpy.array([[[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [0, 0, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [0, 1, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [1, 0, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 1, 0, ],
                                                        [1, 0, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [0, 0, 0, ]]])))
        self.assertTrue(numpy.array_equal(self.nnet.fit_transform('B[20];W[21];B[11]', ChessType.BLACK),
                                          numpy.array([[[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [1, 0, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 1, 0, ],
                                                        [1, 0, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [0, 0, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [0, 1, 0, ]],

                                                       [[1, 1, 1, ],
                                                        [1, 1, 1, ],
                                                        [1, 1, 1, ]]])))

    def test_fit_transform1(self):
        self.args.history_num = 1
        self.assertTrue(numpy.array_equal(self.nnet.fit_transform('B[20];W[21];B[11]', ChessType.WHITE),
                                          numpy.array([[[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [0, 1, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 1, 0, ],
                                                        [1, 0, 0, ]],

                                                       [[0, 0, 0, ],
                                                        [0, 0, 0, ],
                                                        [0, 0, 0, ]]])))


if __name__ == '__main__':
    unittest.main()
