#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Tests for the abstract Game base class in alphazero/game.py."""

import unittest

from alphazero.game import Game


class TestGameDRAWSentinel(unittest.TestCase):
    def test_draw_is_string(self):
        self.assertIsInstance(Game.DRAW, str)

    def test_draw_value(self):
        self.assertEqual(Game.DRAW, "draw")

    def test_draw_accessible_from_instance(self):
        game = Game()
        self.assertEqual(game.DRAW, "draw")


class TestGameComputeReward(unittest.TestCase):
    def setUp(self):
        self.game = Game()

    def test_win_returns_positive_one(self):
        self.assertEqual(self.game.compute_reward("player1", "player1"), 1)

    def test_loss_returns_negative_one(self):
        self.assertEqual(self.game.compute_reward("player1", "player2"), -1)

    def test_draw_returns_zero(self):
        self.assertEqual(self.game.compute_reward(Game.DRAW, "player1"), 0)

    def test_draw_returns_zero_for_any_player(self):
        self.assertEqual(self.game.compute_reward(Game.DRAW, "X"), 0)
        self.assertEqual(self.game.compute_reward(Game.DRAW, "O"), 0)


class TestGameLogStatus(unittest.TestCase):
    def test_log_status_is_noop(self):
        """Default log_status should not raise and return None."""
        game = Game()
        result = game.log_status("board", [1, 2], [0, 1])
        self.assertIsNone(result)


class TestGameAugmentSamples(unittest.TestCase):
    def test_returns_samples_unchanged(self):
        """Default augment_samples should return the input unchanged."""
        game = Game()
        samples = [("board1", [0.5, 0.5], 1), ("board2", [0.3, 0.7], -1)]
        result = game.augment_samples(samples)
        self.assertEqual(result, samples)

    def test_returns_empty_list_unchanged(self):
        game = Game()
        self.assertEqual(game.augment_samples([]), [])


class TestGameAbstractMethods(unittest.TestCase):
    def setUp(self):
        self.game = Game()

    def test_get_initial_state_raises(self):
        with self.assertRaises(NotImplementedError):
            self.game.get_initial_state()

    def test_next_state_raises(self):
        with self.assertRaises(NotImplementedError):
            self.game.next_state("board", 0, "player")

    def test_is_terminal_state_raises(self):
        with self.assertRaises(NotImplementedError):
            self.game.is_terminal_state("board", 0, "player")

    def test_available_actions_raises(self):
        with self.assertRaises(NotImplementedError):
            self.game.available_actions("board")

    def test_get_canonical_form_raises(self):
        with self.assertRaises(NotImplementedError):
            self.game.get_canonical_form("board", "player")


if __name__ == "__main__":
    unittest.main()
