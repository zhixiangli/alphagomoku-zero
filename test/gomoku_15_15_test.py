#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from gomoku_15_15.game import GomokuGame, ChessType


class TestGomoku15x15(unittest.TestCase):
    """Tests for the 15×15 Gomoku game implementation."""

    def setUp(self):
        self.args = dotdict(
            {
                "rows": 15,
                "columns": 15,
                "n_in_row": 5,
            }
        )
        self.game = GomokuGame(self.args)

    def _make_board(self, moves):
        """Build a board string from a list of (player, row, col) tuples."""
        stones = ["%s[%x%x]" % (p, r, c) for p, r, c in moves]
        return ";".join(stones)

    def test_initial_state(self):
        board, player = self.game.get_initial_state()
        self.assertEqual(board, "")
        self.assertEqual(player, ChessType.BLACK)

    def test_available_actions_empty_board(self):
        actions = self.game.available_actions("")
        self.assertEqual(len(actions), 225)
        self.assertEqual(actions, list(range(225)))

    def test_next_state_center(self):
        """Place a stone at center (7,7) of 15×15 board."""
        board, player = self.game.get_initial_state()
        action = 7 * 15 + 7  # (7,7)
        next_board, next_player = self.game.next_state(board, action, player)
        self.assertEqual(next_board, "B[77]")
        self.assertEqual(next_player, ChessType.WHITE)

    def test_next_state_hex_encoding(self):
        """Verify hex encoding for coordinates >= 10 (a-e in hex)."""
        board, player = self.game.get_initial_state()
        # Place at (10, 14) → hex (a, e)
        action = 10 * 15 + 14
        next_board, next_player = self.game.next_state(board, action, player)
        self.assertEqual(next_board, "B[ae]")
        self.assertEqual(next_player, ChessType.WHITE)

    def test_next_state_corner(self):
        """Place at (14, 14) → hex (e, e)."""
        board, player = self.game.get_initial_state()
        action = 14 * 15 + 14
        next_board, next_player = self.game.next_state(board, action, player)
        self.assertEqual(next_board, "B[ee]")
        self.assertEqual(next_player, ChessType.WHITE)

    def test_hex_action_high_coords(self):
        """hex_action correctly encodes coordinates 10-14 as a-e."""
        # (10, 11) → action = 10*15+11 = 161
        self.assertEqual(self.game.hex_action(161), "[ab]")
        # (14, 14) → action = 14*15+14 = 224
        self.assertEqual(self.game.hex_action(224), "[ee]")
        # (0, 0)
        self.assertEqual(self.game.hex_action(0), "[00]")

    def test_dec_action_high_coords(self):
        """dec_action correctly decodes hex coordinates a-e."""
        self.assertEqual(self.game.dec_action("B[ae]"), (10, 14))
        self.assertEqual(self.game.dec_action("W[ee]"), (14, 14))
        self.assertEqual(self.game.dec_action("B[00]"), (0, 0))

    def test_horizontal_win(self):
        """Five consecutive horizontal stones on 15×15 board."""
        moves = [
            (ChessType.BLACK, 7, 3),
            (ChessType.BLACK, 7, 4),
            (ChessType.BLACK, 7, 5),
            (ChessType.BLACK, 7, 6),
            (ChessType.BLACK, 7, 7),
        ]
        board = self._make_board(moves)
        action = 7 * 15 + 7
        result = self.game.is_terminal_state(board, action, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_vertical_win(self):
        """Five consecutive vertical stones on 15×15 board."""
        moves = [
            (ChessType.WHITE, 5, 10),
            (ChessType.WHITE, 6, 10),
            (ChessType.WHITE, 7, 10),
            (ChessType.WHITE, 8, 10),
            (ChessType.WHITE, 9, 10),
        ]
        board = self._make_board(moves)
        action = 9 * 15 + 10
        result = self.game.is_terminal_state(board, action, ChessType.WHITE)
        self.assertEqual(result, ChessType.WHITE)

    def test_diagonal_win(self):
        """Five consecutive diagonal stones on 15×15 board."""
        moves = [
            (ChessType.BLACK, 10, 10),
            (ChessType.BLACK, 11, 11),
            (ChessType.BLACK, 12, 12),
            (ChessType.BLACK, 13, 13),
            (ChessType.BLACK, 14, 14),
        ]
        board = self._make_board(moves)
        action = 14 * 15 + 14
        result = self.game.is_terminal_state(board, action, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_anti_diagonal_win(self):
        """Five consecutive anti-diagonal stones on 15×15 board."""
        moves = [
            (ChessType.BLACK, 0, 14),
            (ChessType.BLACK, 1, 13),
            (ChessType.BLACK, 2, 12),
            (ChessType.BLACK, 3, 11),
            (ChessType.BLACK, 4, 10),
        ]
        board = self._make_board(moves)
        action = 4 * 15 + 10
        result = self.game.is_terminal_state(board, action, ChessType.BLACK)
        self.assertEqual(result, ChessType.BLACK)

    def test_four_in_a_row_not_terminal(self):
        """Four in a row should NOT be terminal when n_in_row=5."""
        moves = [
            (ChessType.BLACK, 0, 0),
            (ChessType.BLACK, 0, 1),
            (ChessType.BLACK, 0, 2),
            (ChessType.BLACK, 0, 3),
        ]
        board = self._make_board(moves)
        result = self.game.is_terminal_state(board, 3, ChessType.BLACK)
        self.assertIsNone(result)

    def test_win_at_board_edge(self):
        """Win along the bottom edge (row 14)."""
        moves = [
            (ChessType.WHITE, 14, 0),
            (ChessType.WHITE, 14, 1),
            (ChessType.WHITE, 14, 2),
            (ChessType.WHITE, 14, 3),
            (ChessType.WHITE, 14, 4),
        ]
        board = self._make_board(moves)
        action = 14 * 15 + 4
        result = self.game.is_terminal_state(board, action, ChessType.WHITE)
        self.assertEqual(result, ChessType.WHITE)

    def test_get_canonical_form(self):
        """Canonical form has correct shape for 15×15 board."""
        board = "B[77];W[78]"
        canonical = self.game.get_canonical_form(board, ChessType.BLACK)
        self.assertEqual(canonical.shape, (15, 15, 2))
        self.assertEqual(canonical[7, 7, 0], 1)  # Black stone, current player
        self.assertEqual(canonical[7, 8, 1], 1)  # White stone, opponent
        self.assertEqual(canonical[7, 7, 1], 0)
        self.assertEqual(canonical[7, 8, 0], 0)

    def test_get_canonical_form_empty(self):
        canonical = self.game.get_canonical_form("", ChessType.BLACK)
        self.assertTrue(numpy.array_equal(canonical, numpy.zeros((15, 15, 2))))

    def test_get_canonical_form_high_coords(self):
        """Canonical form works with hex coordinates a-e."""
        board = "B[ae];W[ea]"
        canonical = self.game.get_canonical_form(board, ChessType.BLACK)
        self.assertEqual(canonical[10, 14, 0], 1)  # B at (10,14)
        self.assertEqual(canonical[14, 10, 1], 1)  # W at (14,10)

    def test_available_actions_after_moves(self):
        board = "B[77];W[78]"
        actions = self.game.available_actions(board)
        self.assertEqual(len(actions), 223)
        self.assertNotIn(7 * 15 + 7, actions)
        self.assertNotIn(7 * 15 + 8, actions)

    def test_augment_samples(self):
        board = numpy.zeros((15, 15, 2))
        board[0, 0, 0] = 1
        policy = numpy.zeros(225)
        policy[0] = 1.0
        samples = [(board, policy, 1.0)]
        augmented = self.game.augment_samples(samples)
        self.assertEqual(len(augmented), 8)
        for b, p, v in augmented:
            self.assertEqual(b.shape, (15, 15, 2))
            self.assertEqual(len(p), 225)
            self.assertAlmostEqual(numpy.sum(p), 1.0)
            self.assertEqual(v, 1.0)

    def test_structure_sgf_high_coords(self):
        sgf = "B[ae];W[ea];B[00]"
        result = self.game.structure_sgf(sgf)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("B", (10, 14)))
        self.assertEqual(result[1], ("W", (14, 10)))
        self.assertEqual(result[2], ("B", (0, 0)))

    def test_to_board(self):
        sgf = "B[00];W[ee]"
        board = self.game.to_board(sgf)
        self.assertEqual(board[0, 0], ChessType.BLACK)
        self.assertEqual(board[14, 14], ChessType.WHITE)
        self.assertEqual(board[7, 7], ChessType.EMPTY)
        self.assertEqual(board.shape, (15, 15))

    def test_empty_board_not_terminal(self):
        result = self.game.is_terminal_state("", 0, ChessType.BLACK)
        self.assertIsNone(result)

    def test_compute_reward_consistency(self):
        self.assertEqual(self.game.compute_reward(ChessType.BLACK, ChessType.BLACK), 1)
        self.assertEqual(self.game.compute_reward(ChessType.BLACK, ChessType.WHITE), -1)
        self.assertEqual(self.game.compute_reward(Game.DRAW, ChessType.BLACK), 0)


class TestGomoku15x15Config(unittest.TestCase):
    """Tests for the 15×15 Gomoku config."""

    def test_config_defaults(self):
        from gomoku_15_15.config import GomokuConfig

        config = GomokuConfig()
        self.assertEqual(config.rows, 15)
        self.assertEqual(config.columns, 15)
        self.assertEqual(config.n_in_row, 5)
        self.assertEqual(config.action_space_size, 225)
        self.assertEqual(config.temp_step, 8)
        self.assertEqual(config.dirichlet_alpha, 0.05)
        self.assertEqual(config.dirichlet_epsilon, 0.10)

    def test_config_paths(self):
        from gomoku_15_15.config import GomokuConfig

        config = GomokuConfig()
        self.assertIn("gomoku_15_15", config.save_checkpoint_path)
        self.assertIn("gomoku_15_15", config.sample_pool_file)


class TestGomoku15x15Trainer(unittest.TestCase):
    """Tests for the 15×15 Gomoku trainer module."""

    def test_main_is_importable(self):
        from gomoku_15_15.trainer import main

        self.assertTrue(callable(main))

    def test_configure_module_importable(self):
        from gomoku_15_15 import configure_module

        self.assertTrue(callable(configure_module))


if __name__ == "__main__":
    unittest.main()
