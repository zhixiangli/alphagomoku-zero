#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import os
import tempfile
import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from connect4.game import Connect4Game, ChessType
from alphazero.nnet import AlphaZeroNNet


class TestConnect4(unittest.TestCase):
    def setUp(self):
        self.args = dotdict(
            {
                "rows": 6,
                "columns": 7,
                "n_in_row": 4,
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
        self.game = Connect4Game(self.args)
        self.nnet = AlphaZeroNNet(self.game, self.args)

    def test_next_player(self):
        self.assertEqual(self.game.next_player(ChessType.BLACK), ChessType.WHITE)
        self.assertEqual(self.game.next_player(ChessType.WHITE), ChessType.BLACK)

    def test_next_state_gravity(self):
        """Pieces should fall to the bottom of the column."""
        board, player = self.game.get_initial_state()

        # First piece in column 3 should land at row 5 (bottom)
        next_board, next_player = self.game.next_state(board, 3, player)
        self.assertEqual(next_player, ChessType.WHITE)
        self.assertEqual(next_board, "B[53]")

        # Second piece in column 3 should land at row 4
        next_board, next_player = self.game.next_state(next_board, 3, next_player)
        self.assertEqual(next_player, ChessType.BLACK)
        self.assertEqual(next_board, "B[53];W[43]")

        # Piece in column 0 should land at row 5
        next_board, next_player = self.game.next_state(next_board, 0, next_player)
        self.assertEqual(next_player, ChessType.WHITE)
        self.assertEqual(next_board, "B[53];W[43];B[50]")

    def test_is_terminal_state_horizontal_win(self):
        # Black wins with 4 in a row horizontally at bottom row
        # B drops at columns 0, 1, 2, 3 (all at row 5)
        sgf = "B[53];B[54];B[55];B[56]"
        self.assertEqual(
            self.game.is_terminal_state(sgf, 6, ChessType.BLACK), ChessType.BLACK
        )

    def test_is_terminal_state_vertical_win(self):
        # Black wins with 4 in a row vertically in column 0
        sgf = "B[50];B[40];B[30];B[20]"
        self.assertEqual(
            self.game.is_terminal_state(sgf, 0, ChessType.BLACK), ChessType.BLACK
        )

    def test_is_terminal_state_diagonal_win(self):
        # Black wins with diagonal: (5,0), (4,1), (3,2), (2,3)
        sgf = "B[50];B[41];B[32];B[23]"
        self.assertEqual(
            self.game.is_terminal_state(sgf, 3, ChessType.BLACK), ChessType.BLACK
        )

    def test_is_terminal_state_anti_diagonal_win(self):
        # Black wins with anti-diagonal: (2,0), (3,1), (4,2), (5,3)
        sgf = "B[20];B[31];B[42];B[53]"
        self.assertEqual(
            self.game.is_terminal_state(sgf, 3, ChessType.BLACK), ChessType.BLACK
        )

    def test_is_terminal_state_not_over(self):
        sgf = "B[50];W[51]"
        self.assertIsNone(self.game.is_terminal_state(sgf, 1, ChessType.WHITE))

    def test_is_terminal_state_draw(self):
        # Fill the entire 6x7 board with no winner
        # Pattern: cell colour depends on (col + row // 2) % 2
        # This gives max 2 consecutive in every direction
        stones = []
        for col in range(7):
            for row in range(5, -1, -1):
                if (col + row // 2) % 2 == 0:
                    color = ChessType.BLACK
                else:
                    color = ChessType.WHITE
                stones.append("%s[%x%x]" % (color, row, col))
        sgf = ";".join(stones)
        last_stone = stones[-1]
        last_col = int(last_stone[3], 16)
        result = self.game.is_terminal_state(sgf, last_col, last_stone[0])
        self.assertEqual(result, Game.DRAW)

    def test_available_actions_empty_board(self):
        board, _ = self.game.get_initial_state()
        self.assertEqual(self.game.available_actions(board), list(range(7)))

    def test_available_actions_partial(self):
        # Fill column 0 completely (6 rows)
        stones = []
        for row in range(5, -1, -1):
            color = ChessType.BLACK if row % 2 == 1 else ChessType.WHITE
            stones.append("%s[%x0]" % (color, row))
        sgf = ";".join(stones)
        self.assertEqual(self.game.available_actions(sgf), [1, 2, 3, 4, 5, 6])

    def test_available_actions_some_moves(self):
        sgf = "B[50];W[51]"
        self.assertEqual(self.game.available_actions(sgf), list(range(7)))

    def test_get_canonical_form(self):
        # Board 'B[53];W[43]' with canonical form for BLACK player
        # Channel 0 (current/BLACK player): B[53] -> (5,3)
        # Channel 1 (opponent/WHITE): W[43] -> (4,3)
        canonical = self.game.get_canonical_form("B[53];W[43]", ChessType.BLACK)
        expected = numpy.zeros((6, 7, 2))
        expected[5, 3, 0] = 1  # Black piece
        expected[4, 3, 1] = 1  # White piece
        self.assertTrue(numpy.array_equal(canonical, expected))

        # Canonical form for WHITE player (swap channels)
        canonical = self.game.get_canonical_form("B[53];W[43]", ChessType.WHITE)
        expected = numpy.zeros((6, 7, 2))
        expected[4, 3, 0] = 1  # White piece (now current player)
        expected[5, 3, 1] = 1  # Black piece (now opponent)
        self.assertTrue(numpy.array_equal(canonical, expected))

    def test_get_canonical_form_empty(self):
        self.assertTrue(
            numpy.array_equal(
                self.game.get_canonical_form("", ChessType.BLACK),
                numpy.zeros((6, 7, 2)),
            )
        )

    def test_value_head_shape(self):
        """Value head must have Flatten→Dense(256)→Dense(1) per AlphaZero paper."""
        model = self.nnet.model
        self.assertEqual(model.value_fc1.out_features, 256)
        self.assertEqual(model.value_fc2.out_features, 1)

    def test_checkpoint_round_trip(self):
        """Checkpoint save and load should preserve model weights."""
        tmpdir = tempfile.mkdtemp()
        prefix = os.path.join(tmpdir, "test_connect4_ckpt")

        self.nnet.save_checkpoint(prefix)
        files = glob.glob(prefix + "*.pt")
        self.assertEqual(len(files), 1)

        # Loading should not raise
        self.nnet.load_checkpoint(prefix)

        # Clean up
        for f in files:
            os.remove(f)
        os.rmdir(tmpdir)

    def test_checkpoint_load_missing(self):
        """load_checkpoint with no matching files should not raise."""
        tmpdir = tempfile.mkdtemp()
        self.nnet.load_checkpoint(os.path.join(tmpdir, "nonexistent_model_prefix"))
        os.rmdir(tmpdir)

    def test_augment_samples(self):
        """augment_samples should double the samples via horizontal flip."""
        board = numpy.zeros((6, 7, 2))
        board[5, 0, 0] = 1  # Black piece at (5, 0)
        policy = numpy.zeros(42)
        policy[0] = 0.5
        policy[3] = 0.5

        augmented = self.game.augment_samples([(board, policy, 1.0)])
        self.assertEqual(len(augmented), 2)

        # Original sample
        self.assertTrue(numpy.array_equal(augmented[0][0], board))
        self.assertTrue(numpy.array_equal(augmented[0][1], policy))

        # Flipped sample: piece at (5, 0) -> (5, 6)
        flipped_board = augmented[1][0]
        self.assertEqual(flipped_board[5, 6, 0], 1)
        self.assertEqual(flipped_board[5, 0, 0], 0)

        # Flipped policy: col 0 -> col 6, col 3 -> col 3
        flipped_policy = augmented[1][1]
        self.assertAlmostEqual(flipped_policy[6], 0.5)
        self.assertAlmostEqual(flipped_policy[3], 0.5)
        self.assertAlmostEqual(flipped_policy[0], 0.0)


if __name__ == "__main__":
    unittest.main()
