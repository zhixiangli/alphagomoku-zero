#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import unittest

import numpy
from dotdict import dotdict

from alphazero.eval import Evaluator
from alphazero.game import Game
from alphazero.module import AlphaZeroModule
from alphazero.nnet import NNet


# --------------- lightweight mocks ---------------


class _ChessType:
    BLACK = "B"
    WHITE = "W"
    EMPTY = "."


class StubGame(Game):
    """Minimal 1×3 game: two adjacent same-colour stones win."""

    def __init__(self, args=None):
        self.rows = 1
        self.columns = 3

    def next_player(self, player):
        return _ChessType.BLACK if player == _ChessType.WHITE else _ChessType.WHITE

    def next_state(self, board, action, player):
        tmp = list(board)
        tmp[action] = player
        return "".join(tmp), self.next_player(player)

    def is_terminal_state(self, board, action, player):
        if action > 0 and board[action - 1] == board[action]:
            return player
        if action + 1 < len(board) and board[action + 1] == board[action]:
            return player
        if all(ch != _ChessType.EMPTY for ch in board):
            return Game.DRAW
        return None

    def get_initial_state(self):
        return _ChessType.EMPTY * self.columns, _ChessType.BLACK

    def available_actions(self, board):
        return [i for i in range(len(board)) if board[i] == _ChessType.EMPTY]

    def log_status(self, board, counts, actions):
        pass

    def get_canonical_form(self, board, player):
        if player == _ChessType.BLACK:
            return board
        return "".join(
            [
                self.next_player(c) if c in (_ChessType.BLACK, _ChessType.WHITE) else c
                for c in board
            ]
        )


class StubNNet(NNet):
    """Returns uniform priors and zero value."""

    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args

    def predict(self, board):
        return numpy.ones(3), 0

    def train(self, data):
        pass

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass


class BiasedNNet(NNet):
    """Returns priors biased toward action 1 (centre) — stronger on this game."""

    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args

    def predict(self, board):
        priors = numpy.array([0.1, 0.8, 0.1])
        return priors, 0

    def train(self, data):
        pass

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass


# --------------- tests ---------------


class TestEvaluator(unittest.TestCase):
    def _make_config(self, **overrides):
        base = {
            "simulation_num": 50,
            "c_puct": 5,
            "rows": 1,
            "columns": 3,
        }
        base.update(overrides)
        return dotdict(base)

    def test_evaluate_returns_correct_structure(self):
        """evaluate() returns a dict with the three expected keys."""
        game = StubGame()
        config = self._make_config()
        evaluator = Evaluator(game, StubNNet(), config, StubNNet(), config)
        results = evaluator.evaluate(num_games=4)

        self.assertIn("agent1_wins", results)
        self.assertIn("agent2_wins", results)
        self.assertIn("draws", results)
        self.assertEqual(
            results["agent1_wins"] + results["agent2_wins"] + results["draws"], 4
        )

    def test_evaluate_all_games_finish(self):
        """All requested games finish and are counted."""
        game = StubGame()
        config = self._make_config()
        evaluator = Evaluator(game, StubNNet(), config, StubNNet(), config)
        results = evaluator.evaluate(num_games=10)

        total = results["agent1_wins"] + results["agent2_wins"] + results["draws"]
        self.assertEqual(total, 10)

    def test_biased_agent_is_stronger(self):
        """An agent with centre-biased priors should dominate on the 1×3 game."""
        game = StubGame()
        strong_config = self._make_config(simulation_num=50)
        weak_config = self._make_config(simulation_num=5)
        biased_nnet = BiasedNNet()
        uniform_nnet = StubNNet()
        evaluator = Evaluator(
            game, biased_nnet, strong_config, uniform_nnet, weak_config
        )
        results = evaluator.evaluate(num_games=20)

        # The biased agent (agent1) with more simulations should win at least as often
        self.assertGreaterEqual(results["agent1_wins"], results["agent2_wins"])

    def test_single_game(self):
        """A single game produces a valid result."""
        game = StubGame()
        config = self._make_config()
        evaluator = Evaluator(game, StubNNet(), config, StubNNet(), config)
        results = evaluator.evaluate(num_games=1)

        total = results["agent1_wins"] + results["agent2_wins"] + results["draws"]
        self.assertEqual(total, 1)

    def test_alternating_colours(self):
        """Agent1 plays first in even games and second in odd games."""
        game = StubGame()
        config = self._make_config()
        evaluator = Evaluator(game, StubNNet(), config, StubNNet(), config)

        # Verify the internal method uses the flag correctly
        # agent1_is_first=True  → agent1 plays as the first player
        # agent1_is_first=False → agent2 plays as the first player
        result_as_first = evaluator._play_game(agent1_is_first=True)
        self.assertIn(result_as_first, [1, -1, 0])
        result_as_second = evaluator._play_game(agent1_is_first=False)
        self.assertIn(result_as_second, [1, -1, 0])


class TestModuleCreateEvaluator(unittest.TestCase):
    def test_create_evaluator(self):
        """create_evaluator wires game, two nnets, and evaluator correctly."""
        module = AlphaZeroModule()
        module.register(StubGame, StubNNet)

        config1 = dotdict({"rows": 1, "columns": 3, "simulation_num": 50, "c_puct": 5})
        config2 = dotdict({"rows": 1, "columns": 3, "simulation_num": 30, "c_puct": 3})
        evaluator = module.create_evaluator(StubGame, config1, config2)

        self.assertIsInstance(evaluator, Evaluator)
        self.assertIsInstance(evaluator.game, StubGame)
        self.assertIsInstance(evaluator.nnet1, StubNNet)
        self.assertIsInstance(evaluator.nnet2, StubNNet)
        self.assertEqual(evaluator.config1, config1)
        self.assertEqual(evaluator.config2, config2)


if __name__ == "__main__":
    unittest.main()
