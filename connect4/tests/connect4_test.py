#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import numpy
from dotdict import dotdict

from alphazero.game import Game
from alphazero.module import AlphaZeroModule
from alphazero.nnet import AlphaZeroNNet
from connect4 import configure_module
from connect4.config import Connect4Config
from connect4.game import Connect4Game, ChessType


class TestConnect4Game(unittest.TestCase):
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

    def test_drop_gravity_in_same_column(self):
        board, player = self.game.get_initial_state()

        board, player = self.game.next_state(board, 0, player)
        self.assertEqual(board, "B[50]")

        board, player = self.game.next_state(board, 0, player)
        self.assertEqual(board, "B[50];W[40]")

    def test_available_actions_one_per_column(self):
        actions = self.game.available_actions("")
        self.assertEqual(actions, [35, 36, 37, 38, 39, 40, 41])

    def test_available_actions_after_partial_fill(self):
        board = "B[50];W[40];B[30]"
        actions = self.game.available_actions(board)
        self.assertEqual(actions[0], 14)

    def test_terminal_vertical(self):
        board = "B[50];B[40];B[30];B[20]"
        self.assertEqual(self.game.is_terminal_state(board, 14, ChessType.BLACK), "B")

    def test_terminal_horizontal(self):
        board = "B[50];B[51];B[52];B[53]"
        self.assertEqual(self.game.is_terminal_state(board, 38, ChessType.BLACK), "B")

    def test_terminal_diagonal(self):
        board = "B[50];B[41];B[32];B[23]"
        self.assertEqual(self.game.is_terminal_state(board, 17, ChessType.BLACK), "B")

    def test_draw(self):
        args = dotdict({"rows": 2, "columns": 2, "n_in_row": 3})
        game = Connect4Game(args)
        board = "B[10];W[11];B[00];W[01]"
        self.assertEqual(game.is_terminal_state(board, 1, ChessType.WHITE), Game.DRAW)

    def test_get_canonical_form(self):
        board = "B[50];W[51];B[41]"
        canonical = self.game.get_canonical_form(board, ChessType.WHITE)
        self.assertEqual(canonical[5, 1, 0], 1)
        self.assertEqual(canonical[5, 0, 1], 1)
        self.assertEqual(canonical[4, 1, 1], 1)

    def test_augment_samples(self):
        board = numpy.zeros((6, 7, 2))
        board[5, 0, 0] = 1
        policy = numpy.zeros(42)
        policy[35] = 1
        augmented = self.game.augment_samples([(board, policy, 1)])
        self.assertEqual(len(augmented), 2)
        _, flipped_policy, _ = augmented[1]
        self.assertEqual(flipped_policy[41], 1)


class TestConnect4ModuleIntegration(unittest.TestCase):
    def test_configure_module_registers_connect4(self):
        module = AlphaZeroModule()
        configure_module(module)
        self.assertIs(module.resolve_nnet_class(Connect4Game), AlphaZeroNNet)

    @patch("connect4.trainer.run_training")
    @patch("connect4.trainer.setup_logging")
    def test_trainer_main_wires(self, _mock_logging, mock_run):
        from connect4.trainer import main

        with patch("sys.argv", ["trainer", "-rows", "6", "-columns", "7"]):
            main()
        self.assertEqual(mock_run.call_count, 1)
        args, _ = mock_run.call_args
        config = args[2]
        self.assertIsInstance(config, Connect4Config)
        self.assertEqual(config.rows, 6)
        self.assertEqual(config.columns, 7)
        self.assertEqual(config.batch_size, 256)


if __name__ == "__main__":
    unittest.main()
