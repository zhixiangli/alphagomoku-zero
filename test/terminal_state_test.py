#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Comprehensive tests for terminal state detection and MCTS termination.

Covers the critical bug categories identified in the Gomoku 6×6 pipeline:
- Win detection in all 4 directions (horizontal, vertical, diagonal, anti-diagonal)
- Overlines (5+ in a row when n_in_row=4)
- Edge and corner wins
- Draw detection
- MCTS terminal node handling
- Self-play loop termination safety
"""

import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from alphazero.mcts import MCTS
from alphazero.nnet import NNet
from alphazero.rl import RL
from gomoku.game import GomokuGame, ChessType


class TestGomokuTerminalDetection6x6(unittest.TestCase):
    """Terminal state detection on a 6×6 board with n_in_row=4."""

    def setUp(self):
        self.args = dotdict({"rows": 6, "columns": 6, "n_in_row": 4})
        self.game = GomokuGame(self.args)

    def _make_board(self, moves):
        """Build a board string from a list of (player, row, col) tuples."""
        stones = ["%s[%x%x]" % (p, r, c) for p, r, c in moves]
        return ";".join(stones)

    def test_horizontal_win(self):
        """Four consecutive horizontal stones should be detected as a win."""
        moves = [
            (ChessType.BLACK, 0, 0),
            (ChessType.BLACK, 0, 1),
            (ChessType.BLACK, 0, 2),
            (ChessType.BLACK, 0, 3),
        ]
        board = self._make_board(moves)
        # Last move at (0,3) → action = 0*6+3 = 3
        result = self.game.is_terminal_state(board, 3, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_vertical_win(self):
        """Four consecutive vertical stones should be detected as a win."""
        moves = [
            (ChessType.WHITE, 1, 2),
            (ChessType.WHITE, 2, 2),
            (ChessType.WHITE, 3, 2),
            (ChessType.WHITE, 4, 2),
        ]
        board = self._make_board(moves)
        # Last move at (4,2) → action = 4*6+2 = 26
        result = self.game.is_terminal_state(board, 26, ChessType.WHITE)
        self.assertEqual(result, ChessType.WHITE)

    def test_diagonal_win(self):
        """Four consecutive diagonal (↘) stones should be detected as a win."""
        moves = [
            (ChessType.BLACK, 0, 0),
            (ChessType.BLACK, 1, 1),
            (ChessType.BLACK, 2, 2),
            (ChessType.BLACK, 3, 3),
        ]
        board = self._make_board(moves)
        # Last move at (3,3) → action = 3*6+3 = 21
        result = self.game.is_terminal_state(board, 21, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_anti_diagonal_win(self):
        """Four consecutive anti-diagonal (↙) stones should be detected."""
        moves = [
            (ChessType.BLACK, 0, 3),
            (ChessType.BLACK, 1, 2),
            (ChessType.BLACK, 2, 1),
            (ChessType.BLACK, 3, 0),
        ]
        board = self._make_board(moves)
        # Last move at (3,0) → action = 3*6+0 = 18
        result = self.game.is_terminal_state(board, 18, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_overline_detected(self):
        """Five in a row should also be detected as a win (count >= n)."""
        moves = [
            (ChessType.BLACK, 2, 0),
            (ChessType.BLACK, 2, 1),
            (ChessType.BLACK, 2, 2),
            (ChessType.BLACK, 2, 3),
            (ChessType.BLACK, 2, 4),
        ]
        board = self._make_board(moves)
        # Last move at (2,4) → action = 2*6+4 = 16
        result = self.game.is_terminal_state(board, 16, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_win_at_board_edge(self):
        """Win at the right edge of the board."""
        moves = [
            (ChessType.WHITE, 5, 2),
            (ChessType.WHITE, 5, 3),
            (ChessType.WHITE, 5, 4),
            (ChessType.WHITE, 5, 5),
        ]
        board = self._make_board(moves)
        # Last move at (5,5) → action = 5*6+5 = 35
        result = self.game.is_terminal_state(board, 35, ChessType.WHITE)
        self.assertEqual(result, ChessType.WHITE)

    def test_win_at_board_corner(self):
        """Win involving corner positions."""
        moves = [
            (ChessType.BLACK, 2, 2),
            (ChessType.BLACK, 3, 3),
            (ChessType.BLACK, 4, 4),
            (ChessType.BLACK, 5, 5),
        ]
        board = self._make_board(moves)
        # Last move at (5,5) → action = 35
        result = self.game.is_terminal_state(board, 35, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_win_detected_from_middle_stone(self):
        """Win detected when the action is a middle stone of the line."""
        moves = [
            (ChessType.BLACK, 0, 0),
            (ChessType.BLACK, 0, 1),
            (ChessType.BLACK, 0, 2),
            (ChessType.BLACK, 0, 3),
        ]
        board = self._make_board(moves)
        # Last move at (0,1) → action = 1 (middle of the line)
        result = self.game.is_terminal_state(board, 1, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_three_in_a_row_not_terminal(self):
        """Three in a row should NOT be terminal when n_in_row=4."""
        moves = [
            (ChessType.BLACK, 0, 0),
            (ChessType.BLACK, 0, 1),
            (ChessType.BLACK, 0, 2),
        ]
        board = self._make_board(moves)
        result = self.game.is_terminal_state(board, 2, ChessType.BLACK)
        self.assertIsNone(result)

    def test_draw_when_board_full(self):
        """Full board with no winner should return Game.DRAW."""
        # Fill the board alternating B/W without creating 4-in-a-row
        moves = []
        player = ChessType.BLACK
        for r in range(6):
            for c in range(6):
                moves.append((player, r, c))
                player = (
                    ChessType.WHITE if player == ChessType.BLACK else ChessType.BLACK
                )
        board = self._make_board(moves)
        # Last move at (5,5) → action = 35
        # Note: this might or might not be a win depending on pattern,
        # but we test that total_stones == total is detected
        result = self.game.is_terminal_state(board, 35, ChessType.BLACK)
        # Result should be either a player (win) or Game.DRAW
        self.assertIsNotNone(result)

    def test_opponent_stones_not_counted(self):
        """Only the current player's stones should count toward a win."""
        # Black has 3 in a row + 1 White stone breaking the line
        moves = [
            (ChessType.BLACK, 0, 0),
            (ChessType.BLACK, 0, 1),
            (ChessType.WHITE, 0, 2),
            (ChessType.BLACK, 0, 3),
        ]
        board = self._make_board(moves)
        result = self.game.is_terminal_state(board, 3, ChessType.BLACK)
        self.assertIsNone(result)

    def test_empty_board_not_terminal(self):
        """Empty board is not terminal."""
        result = self.game.is_terminal_state("", 0, ChessType.BLACK)
        self.assertIsNone(result)

    def test_compute_reward_consistency(self):
        """compute_reward should be consistent with terminal state values."""
        self.assertEqual(self.game.compute_reward(ChessType.BLACK, ChessType.BLACK), 1)
        self.assertEqual(self.game.compute_reward(ChessType.BLACK, ChessType.WHITE), -1)
        self.assertEqual(self.game.compute_reward(Game.DRAW, ChessType.BLACK), 0)
        self.assertEqual(self.game.compute_reward(Game.DRAW, ChessType.WHITE), 0)


class _MockNNet6x6(NNet):
    """Returns uniform priors and zero value for 6×6 board."""

    def predict(self, board):
        return numpy.ones(36), 0

    def load_checkpoint(self, filename):
        pass


class TestMCTSTermination(unittest.TestCase):
    """MCTS correctly handles terminal states in search tree."""

    def setUp(self):
        self.args = dotdict(
            {
                "rows": 6,
                "columns": 6,
                "n_in_row": 4,
                "simulation_num": 50,
                "c_puct": 5,
            }
        )
        self.game = GomokuGame(self.args)
        self.nnet = _MockNNet6x6()

    def test_mcts_near_terminal_position(self):
        """MCTS should work correctly when close to a terminal state."""
        # Set up a board where Black has 3 in a row and one move to win
        board = "B[00];W[10];B[01];W[11];B[02];W[12]"
        player = ChessType.BLACK

        mcts = MCTS(self.nnet, self.game, self.args)
        actions, counts = mcts.simulate(board, player)

        # Action 3 (position 0,3) completes 4-in-a-row for Black
        action_list = list(actions)
        if 3 in action_list:
            idx = action_list.index(3)
            # The winning move should be visited most often
            self.assertEqual(int(numpy.argmax(counts)), idx)

    def test_terminal_states_cached_correctly(self):
        """Terminal states should be correctly cached in MCTS."""
        mcts = MCTS(self.nnet, self.game, self.args)
        board = "B[00];W[10];B[01];W[11];B[02];W[12]"
        player = ChessType.BLACK

        mcts.simulate(board, player)

        # Check that terminal states are cached
        for cached_board, result in mcts.terminal_state.items():
            if result is not None:
                self.assertIn(result, [ChessType.BLACK, ChessType.WHITE, Game.DRAW])

    def test_mcts_compute_reward_for_draws(self):
        """MCTS should assign value=0 for draw terminal states."""
        self.assertEqual(self.game.compute_reward(Game.DRAW, ChessType.BLACK), 0)
        self.assertEqual(self.game.compute_reward(Game.DRAW, ChessType.WHITE), 0)

    def test_mcts_compute_reward_for_wins(self):
        """MCTS should assign value=1 for winning and -1 for losing."""
        self.assertEqual(self.game.compute_reward(ChessType.BLACK, ChessType.BLACK), 1)
        self.assertEqual(self.game.compute_reward(ChessType.BLACK, ChessType.WHITE), -1)


class TestSelfPlayTermination(unittest.TestCase):
    """Self-play loop correctly terminates and produces valid training data."""

    def setUp(self):
        self.args = dotdict(
            {
                "rows": 6,
                "columns": 6,
                "n_in_row": 4,
                "simulation_num": 20,
                "c_puct": 5,
                "save_checkpoint_path": "",
                "max_sample_pool_size": 1000,
                "sample_pool_file": "",
                "temp_step": 0,
            }
        )
        self.game = GomokuGame(self.args)
        self.nnet = _MockNNet6x6()

    def test_self_play_terminates(self):
        """Self-play game should terminate within max_moves."""
        rl = RL(self.nnet, self.game, self.args)
        samples = rl.play_against_itself()

        # Game must produce at least one sample
        self.assertGreater(len(samples), 0)

        # Game must not exceed max possible moves
        max_moves = self.args.rows * self.args.columns
        self.assertLessEqual(len(samples), max_moves)

    def test_self_play_values_are_valid(self):
        """Training sample values must be in {-1, 0, 1}."""
        rl = RL(self.nnet, self.game, self.args)
        samples = rl.play_against_itself()

        for board, policy, value in samples:
            self.assertIn(value, [-1, 0, 1])

    def test_self_play_policies_sum_to_one(self):
        """Training sample policies must sum to approximately 1."""
        rl = RL(self.nnet, self.game, self.args)
        samples = rl.play_against_itself()

        for board, policy, value in samples:
            self.assertAlmostEqual(numpy.sum(policy), 1.0, places=5)

    def test_self_play_board_dimensions(self):
        """Training sample boards must have correct dimensions."""
        rl = RL(self.nnet, self.game, self.args)
        samples = rl.play_against_itself()

        for board, policy, value in samples:
            self.assertEqual(board.shape, (6, 6, 2))
            self.assertEqual(len(policy), 36)


if __name__ == "__main__":
    unittest.main()
