#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import os
import tempfile
import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from gomoku_9_9.game import GomokuGame, ChessType
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
        """Value head must have Flatten→Dense(256)→Dense(1) per AlphaZero paper."""
        model = self.nnet.model
        # Value head should have: value_fc1 (Linear, 256) → value_fc2 (Linear, 1)
        self.assertEqual(model.value_fc1.out_features, 256)
        self.assertEqual(model.value_fc2.out_features, 1)

    def test_checkpoint_round_trip(self):
        """Checkpoint save and load should preserve model weights."""
        tmpdir = tempfile.mkdtemp()
        prefix = os.path.join(tmpdir, "test_gomoku_ckpt")

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

    def test_train_logs_progress(self):
        """train() should log epoch progress with loss and timing info."""
        self.args.batch_size = 4
        self.args.epochs = 2
        nnet = AlphaZeroNNet(self.game, self.args)

        # Create minimal training data
        rows, cols = self.args.rows, self.args.columns
        data = [
            (numpy.zeros((rows, cols, 2)), numpy.ones(rows * cols) / (rows * cols), 1.0)
            for _ in range(8)
        ]

        with self.assertLogs("root", level="INFO") as cm:
            nnet.train(data)

        messages = "\n".join(cm.output)
        self.assertIn("training start: 8 samples, 2 epochs, batch_size=4", messages)
        self.assertIn("epoch 1/2", messages)
        self.assertIn("epoch 2/2", messages)
        self.assertIn("policy_loss:", messages)
        self.assertIn("value_loss:", messages)
        self.assertIn("training complete:", messages)


class TestGomokuHelpers(unittest.TestCase):
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

    def test_hex_action(self):
        self.assertEqual(self.game.hex_action(0), "[00]")
        self.assertEqual(self.game.hex_action(4), "[11]")
        self.assertEqual(self.game.hex_action(8), "[22]")

    def test_dec_action(self):
        self.assertEqual(self.game.dec_action("B[12]"), (1, 2))
        self.assertEqual(self.game.dec_action("W[00]"), (0, 0))
        self.assertEqual(self.game.dec_action("B[22]"), (2, 2))

    def test_structure_sgf(self):
        sgf = "B[00];W[11];B[22]"
        result = self.game.structure_sgf(sgf)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("B", (0, 0)))
        self.assertEqual(result[1], ("W", (1, 1)))
        self.assertEqual(result[2], ("B", (2, 2)))

    def test_structure_sgf_empty(self):
        result = self.game.structure_sgf("")
        self.assertEqual(result, [])

    def test_to_board(self):
        sgf = "B[00];W[11]"
        board = self.game.to_board(sgf)
        self.assertEqual(board[0, 0], ChessType.BLACK)
        self.assertEqual(board[1, 1], ChessType.WHITE)
        self.assertEqual(board[0, 1], ChessType.EMPTY)
        self.assertEqual(board.shape, (3, 3))

    def test_to_board_empty(self):
        board = self.game.to_board("")
        expected = numpy.full((3, 3), ChessType.EMPTY, dtype="U1")
        self.assertTrue(numpy.array_equal(board, expected))

    def test_initial_state(self):
        board, player = self.game.get_initial_state()
        self.assertEqual(board, "")
        self.assertEqual(player, ChessType.BLACK)

    def test_available_actions_empty_board(self):
        actions = self.game.available_actions("")
        self.assertEqual(actions, list(range(9)))

    def test_augment_samples(self):
        board = numpy.zeros((3, 3, 2))
        board[0, 0, 0] = 1
        policy = numpy.zeros(9)
        policy[0] = 1.0
        samples = [(board, policy, 1.0)]
        augmented = self.game.augment_samples(samples)
        self.assertEqual(len(augmented), 8)
        for b, p, v in augmented:
            self.assertEqual(b.shape, (3, 3, 2))
            self.assertEqual(len(p), 9)
            self.assertAlmostEqual(numpy.sum(p), 1.0)
            self.assertEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
