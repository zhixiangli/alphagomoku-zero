#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Tests for battle.py - Command enum, BattleAgent, and GomokuBattleAgent."""

import unittest

import numpy
from dotdict import dotdict
from unittest.mock import patch

from alphazero.nnet import NNet
from battle import Command, BattleAgent, GomokuBattleAgent
from gomoku.game import GomokuGame, ChessType


class TestCommandEnum(unittest.TestCase):
    def test_next_black_value(self):
        self.assertEqual(Command.NEXT_BLACK.value, 1)

    def test_next_white_value(self):
        self.assertEqual(Command.NEXT_WHITE.value, 2)

    def test_next_black_name(self):
        self.assertEqual(Command.NEXT_BLACK.name, "NEXT_BLACK")

    def test_next_white_name(self):
        self.assertEqual(Command.NEXT_WHITE.name, "NEXT_WHITE")

    def test_enum_members(self):
        self.assertEqual(len(Command), 2)


class TestBattleAgent(unittest.TestCase):
    def test_next_raises_not_implemented(self):
        agent = BattleAgent()
        with self.assertRaises(NotImplementedError):
            agent.next("some_sgf", ChessType.BLACK)


class _MockNNet3x3(NNet):
    """Returns uniform priors and zero value for 3×3 board."""

    def predict(self, board):
        return numpy.ones(9), 0

    def load_checkpoint(self, filename):
        pass


class TestGomokuBattleAgent(unittest.TestCase):
    def setUp(self):
        self.args = dotdict(
            {
                "rows": 3,
                "columns": 3,
                "n_in_row": 2,
                "simulation_num": 20,
                "c_puct": 5,
            }
        )
        self.game = GomokuGame(self.args)
        self.nnet = _MockNNet3x3()
        self.agent = GomokuBattleAgent(self.nnet, self.game, self.args)

    def test_next_returns_dict_with_indices(self):
        result = self.agent.next("", ChessType.BLACK)
        self.assertIn("rowIndex", result)
        self.assertIn("columnIndex", result)

    def test_next_returns_valid_position(self):
        result = self.agent.next("", ChessType.BLACK)
        self.assertGreaterEqual(result["rowIndex"], 0)
        self.assertLess(result["rowIndex"], self.args.rows)
        self.assertGreaterEqual(result["columnIndex"], 0)
        self.assertLess(result["columnIndex"], self.args.columns)

    def test_next_with_existing_board(self):
        """Agent should pick a valid move on a partially filled board."""
        sgf = "B[00];W[11]"
        result = self.agent.next(sgf, ChessType.BLACK)
        self.assertIn("rowIndex", result)
        self.assertIn("columnIndex", result)
        # Position should not be already occupied
        occupied = {(0, 0), (1, 1)}
        self.assertNotIn((result["rowIndex"], result["columnIndex"]), occupied)

    def test_agent_has_mcts_instance(self):
        self.assertIsNotNone(self.agent.mcts)

    @patch("sys.stdin")
    @patch("builtins.print")
    def test_start_processes_next_black(self, mock_print, mock_stdin):
        """BattleAgent.start reads JSON commands and responds."""
        import json

        cmd = json.dumps({"command": "NEXT_BLACK", "chessboard": ""})
        # Return the command line then empty to break the loop
        mock_stdin.readline.side_effect = [cmd, KeyboardInterrupt]
        try:
            self.agent.start()
        except KeyboardInterrupt:
            pass
        mock_print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
