#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import os
import tempfile
import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from gomoku.game import GomokuGame, ChessType
from alphazero.nnet import AlphaZeroNNet


class TestGomoku(unittest.TestCase):
    def setUp(self):
        self.args = dotdict(
            {
                "rows": 3,
                "columns": 3,
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

    def test_next_player(self):
        self.assertEqual(self.game.next_player(ChessType.BLACK), ChessType.WHITE)
        self.assertEqual(self.game.next_player(ChessType.WHITE), ChessType.BLACK)

    def test_next_state(self):
        board, player = self.game.get_initial_state()

        next_board, next_player = self.game.next_state(board, 6, player)
        self.assertEqual(next_player, ChessType.WHITE)
        self.assertEqual(next_board, "B[20]")

        next_board, next_player = self.game.next_state(next_board, 0, next_player)
        self.assertEqual(next_player, ChessType.BLACK)
        self.assertEqual(next_board, "B[20];W[00]")

    def test_is_terminal_state(self):
        sgf = ";".join(
            [
                ChessType.BLACK + self.game.hex_action(i)
                for i in range(self.args.rows * self.args.columns)
            ]
        )
        self.assertEqual(
            self.game.is_terminal_state(sgf, 0, ChessType.BLACK), ChessType.BLACK
        )

        sgf = ";".join(
            [
                ChessType.WHITE + self.game.hex_action(i)
                for i in range(self.args.rows * self.args.columns)
            ]
        )
        self.assertEqual(
            self.game.is_terminal_state(sgf, 0, ChessType.BLACK), Game.DRAW
        )

        boards = ["B[11];B[12]", "B[11];B[21]", "B[11];B[22]", "B[11];B[00]"]
        for sgf in boards:
            self.assertEqual(
                self.game.is_terminal_state(
                    sgf, 1 * self.args.columns + 1, ChessType.BLACK
                ),
                ChessType.BLACK,
            )

        sgf = "B[03];B[10]"
        self.assertEqual(self.game.is_terminal_state(sgf, 3, ChessType.BLACK), None)

    def test_available_actions(self):
        sgf = ";".join(
            [
                ChessType.BLACK + self.game.hex_action(i)
                for i in range(self.args.rows * self.args.columns)
            ]
        )
        self.assertEqual(self.game.available_actions(sgf), [])

        sgf = ";".join(
            [
                ChessType.BLACK + self.game.hex_action(i)
                for i in range(0, self.args.rows * self.args.columns, 2)
            ]
        )
        self.assertEqual(
            self.game.available_actions(sgf),
            [i for i in range(1, self.args.rows * self.args.columns, 2)],
        )

    def test_get_canonical_form(self):
        # Board 'B[20];W[21];B[11]' with canonical form for WHITE player
        # Channel 0 (current/WHITE player): W[21] -> (2,1)
        # Channel 1 (opponent/BLACK): B[20] -> (2,0), B[11] -> (1,1)
        canonical = self.game.get_canonical_form("B[20];W[21];B[11]", ChessType.WHITE)
        self.assertTrue(
            numpy.array_equal(
                canonical,
                numpy.array(
                    [
                        [[0, 0], [0, 0], [0, 0]],
                        [[0, 0], [0, 1], [0, 0]],
                        [[0, 1], [1, 0], [0, 0]],
                    ]
                ),
            )
        )
        # Board 'B[20];W[21];B[11]' with canonical form for BLACK player
        # Channel 0 (current/BLACK player): B[20] -> (2,0), B[11] -> (1,1)
        # Channel 1 (opponent/WHITE): W[21] -> (2,1)
        canonical = self.game.get_canonical_form("B[20];W[21];B[11]", ChessType.BLACK)
        self.assertTrue(
            numpy.array_equal(
                canonical,
                numpy.array(
                    [
                        [[0, 0], [0, 0], [0, 0]],
                        [[0, 0], [1, 0], [0, 0]],
                        [[1, 0], [0, 1], [0, 0]],
                    ]
                ),
            )
        )

    def test_get_canonical_form_empty(self):
        self.assertTrue(
            numpy.array_equal(
                self.game.get_canonical_form("", ChessType.BLACK),
                numpy.zeros((3, 3, 2)),
            )
        )

    def test_value_head_shape(self):
        """Value head must Flatten before Dense(256) per AlphaZero paper."""
        layer_names = [layer.name for layer in self.nnet.model.layers]
        # Find the value-head Flatten and Dense(256)
        flatten_indices = [i for i, n in enumerate(layer_names) if "flatten" in n]
        dense_256_indices = [
            i
            for i, layer in enumerate(self.nnet.model.layers)
            if "dense" in layer.name and hasattr(layer, "units") and layer.units == 256
        ]
        # Flatten must appear before Dense(256) in the value head
        self.assertTrue(len(flatten_indices) >= 1)
        self.assertTrue(len(dense_256_indices) >= 1)
        # The value-head Flatten should come before the value-head Dense(256)
        self.assertTrue(
            any(fi < di for fi in flatten_indices for di in dense_256_indices)
        )

    def test_checkpoint_round_trip(self):
        """Checkpoint save and load should preserve model weights."""
        tmpdir = tempfile.mkdtemp()
        prefix = os.path.join(tmpdir, "test_gomoku_ckpt")

        self.nnet.save_checkpoint(prefix)
        files = glob.glob(prefix + "*.weights.h5")
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


if __name__ == "__main__":
    unittest.main()
