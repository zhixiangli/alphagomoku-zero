#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import os
import tempfile
import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from tictactoe.game import TicTacToeGame
from alphazero.nnet import AlphaZeroNNet
from gomoku.game import ChessType


class TestTicTacToe(unittest.TestCase):

    def setUp(self):
        self.args = dotdict({
            'rows': 3,
            'columns': 3,
            'n_in_row': 3,
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
        self.game = TicTacToeGame(self.args)
        self.nnet = AlphaZeroNNet(self.game, self.args)

    def test_next_player(self):
        self.assertEqual(self.game.next_player(ChessType.BLACK), ChessType.WHITE)
        self.assertEqual(self.game.next_player(ChessType.WHITE), ChessType.BLACK)

    def test_next_state(self):
        board, player = self.game.get_initial_state()

        next_board, next_player = self.game.next_state(board, 4, player)
        self.assertEqual(next_player, ChessType.WHITE)
        self.assertEqual(next_board, 'B[11]')

        next_board, next_player = self.game.next_state(next_board, 0, next_player)
        self.assertEqual(next_player, ChessType.BLACK)
        self.assertEqual(next_board, 'B[11];W[00]')

    def test_is_terminal_state_row_win(self):
        # Black wins with top row: (0,0), (0,1), (0,2)
        sgf = 'B[00];B[01];B[02]'
        self.assertEqual(self.game.is_terminal_state(sgf, 2, ChessType.BLACK), ChessType.BLACK)

    def test_is_terminal_state_col_win(self):
        # Black wins with left column: (0,0), (1,0), (2,0)
        sgf = 'B[00];B[10];B[20]'
        self.assertEqual(self.game.is_terminal_state(sgf, 6, ChessType.BLACK), ChessType.BLACK)

    def test_is_terminal_state_diag_win(self):
        # Black wins with diagonal: (0,0), (1,1), (2,2)
        sgf = 'B[00];B[11];B[22]'
        self.assertEqual(self.game.is_terminal_state(sgf, 8, ChessType.BLACK), ChessType.BLACK)

    def test_is_terminal_state_anti_diag_win(self):
        # Black wins with anti-diagonal: (0,2), (1,1), (2,0)
        sgf = 'B[02];B[11];B[20]'
        self.assertEqual(self.game.is_terminal_state(sgf, 6, ChessType.BLACK), ChessType.BLACK)

    def test_is_terminal_state_draw(self):
        # Full board, no winner
        # B W B
        # B B W
        # W B W
        sgf = 'B[00];W[01];B[02];B[10];B[11];W[12];W[20];B[21];W[22]'
        self.assertEqual(self.game.is_terminal_state(sgf, 8, ChessType.WHITE), Game.DRAW)

    def test_is_terminal_state_not_over(self):
        sgf = 'B[00];W[11]'
        self.assertIsNone(self.game.is_terminal_state(sgf, 4, ChessType.WHITE))

    def test_available_actions(self):
        board, _ = self.game.get_initial_state()
        self.assertEqual(self.game.available_actions(board), list(range(9)))

        sgf = 'B[00];W[11]'
        self.assertEqual(self.game.available_actions(sgf), [1, 2, 3, 5, 6, 7, 8])

    def test_get_canonical_form(self):
        # Channel 0 (current/BLACK): B[00] -> (0,0)
        # Channel 1 (opponent/WHITE): W[11] -> (1,1)
        self.assertTrue(numpy.array_equal(
            self.game.get_canonical_form('B[00];W[11]', ChessType.BLACK),
            numpy.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])))
        # Channel 0 (current/WHITE): W[11] -> (1,1)
        # Channel 1 (opponent/BLACK): B[00] -> (0,0)
        self.assertTrue(numpy.array_equal(
            self.game.get_canonical_form('B[00];W[11]', ChessType.WHITE),
            numpy.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                         [[1, 0, 0], [0, 0, 0], [0, 0, 0]]])))

    def test_checkpoint_round_trip(self):
        tmpdir = tempfile.mkdtemp()
        prefix = os.path.join(tmpdir, 'test_tictactoe_ckpt')

        self.nnet.save_checkpoint(prefix)
        files = glob.glob(prefix + '*.weights.h5')
        self.assertEqual(len(files), 1)

        self.nnet.load_checkpoint(prefix)

        for f in files:
            os.remove(f)
        os.rmdir(tmpdir)

    def test_checkpoint_load_missing(self):
        tmpdir = tempfile.mkdtemp()
        self.nnet.load_checkpoint(os.path.join(tmpdir, 'nonexistent_model_prefix'))
        os.rmdir(tmpdir)


if __name__ == '__main__':
    unittest.main()
