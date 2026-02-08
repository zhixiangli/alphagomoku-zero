#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from alphazero.module import AlphaZeroModule
from alphazero.nnet import AlphaZeroNNet, NNet
from alphazero.rl import RL
from gomoku import configure_module
from gomoku.config import GomokuConfig
from gomoku.game import GomokuGame
from gomoku.nnet import GomokuNNet


# --------------- lightweight mocks ---------------

class _ChessType:
    BLACK = 'B'
    WHITE = 'W'
    EMPTY = '.'


class StubGame(Game):
    """Minimal 1×3 game for DI tests (no heavy deps)."""

    def __init__(self, args=None):
        self.rows = 1
        self.columns = 3

    @property
    def action_space_size(self):
        return self.rows * self.columns

    def next_player(self, player):
        return _ChessType.BLACK if player == _ChessType.WHITE else _ChessType.WHITE

    def next_state(self, board, action, player):
        tmp = list(board)
        tmp[action] = player
        return ''.join(tmp), self.next_player(player)

    def is_terminal_state(self, board, action, player):
        if (action > 0 and board[action - 1] == board[action]) or \
           (action == 0 and board[action + 1] == board[action]):
            return player
        if all(ch != _ChessType.EMPTY for ch in board):
            return _ChessType.EMPTY
        return None

    def get_initial_state(self):
        return _ChessType.EMPTY * self.columns, _ChessType.BLACK

    def available_actions(self, board):
        return [i for i in range(len(board)) if board[i] == _ChessType.EMPTY]

    def log_status(self, board, counts, actions):
        pass

    def compute_reward(self, winner, player):
        if winner == _ChessType.EMPTY:
            return 0
        return 1 if player == winner else -1


class StubNNet(NNet):
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def predict(self, data):
        return numpy.array([1] * self.game.columns), 0

    def train(self, data):
        pass

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass


# --------------- tests ---------------

class TestAlphaZeroModule(unittest.TestCase):

    def test_default_nnet_binding(self):
        """Unregistered games fall back to AlphaZeroNNet."""
        module = AlphaZeroModule()
        self.assertIs(module.resolve_nnet_class(StubGame), AlphaZeroNNet)

    def test_register_and_resolve(self):
        """Registered game resolves to its bound NNet class."""
        module = AlphaZeroModule()
        module.register(StubGame, StubNNet)
        self.assertIs(module.resolve_nnet_class(StubGame), StubNNet)

    def test_register_returns_self(self):
        """register() returns the module for fluent chaining."""
        module = AlphaZeroModule()
        self.assertIs(module.register(StubGame, StubNNet), module)

    def test_create_trainer(self):
        """create_trainer wires game, nnet, and RL correctly."""
        module = AlphaZeroModule()
        module.register(StubGame, StubNNet)

        args = dotdict({
            'rows': 1,
            'columns': 3,
            'max_sample_pool_size': 100,
            'sample_pool_file': '',
        })
        trainer = module.create_trainer(StubGame, args)

        self.assertIsInstance(trainer, RL)
        self.assertIsInstance(trainer.game, StubGame)
        self.assertIsInstance(trainer.nnet, StubNNet)

    def test_multiple_registrations(self):
        """Each game class gets its own binding."""
        class AnotherGame(StubGame):
            pass

        class AnotherNNet(StubNNet):
            pass

        module = AlphaZeroModule()
        module.register(StubGame, StubNNet)
        module.register(AnotherGame, AnotherNNet)

        self.assertIs(module.resolve_nnet_class(StubGame), StubNNet)
        self.assertIs(module.resolve_nnet_class(AnotherGame), AnotherNNet)


class TestGomokuModuleIntegration(unittest.TestCase):

    def test_configure_module_registers_gomoku(self):
        """configure_module binds GomokuGame → GomokuNNet."""
        module = AlphaZeroModule()
        configure_module(module)
        self.assertIs(module.resolve_nnet_class(GomokuGame), GomokuNNet)

    def test_create_gomoku_trainer(self):
        """Full Gomoku trainer can be created via DI."""
        module = AlphaZeroModule()
        configure_module(module)
        config = GomokuConfig(rows=3, columns=3, n_in_row=2)
        trainer = module.create_trainer(GomokuGame, config)

        self.assertIsInstance(trainer, RL)
        self.assertIsInstance(trainer.game, GomokuGame)
        self.assertIsInstance(trainer.nnet, GomokuNNet)

    def test_decoupling_game_does_not_import_nnet(self):
        """GomokuGame module has no dependency on GomokuNNet."""
        import gomoku.game as game_module
        with open(game_module.__file__) as f:
            source = f.read()
        self.assertNotIn('GomokuNNet', source)
        self.assertNotIn('nnet', source)

    def test_decoupling_config_does_not_import_nnet_or_game(self):
        """GomokuConfig module has no dependency on game or nnet."""
        import gomoku.config as config_module
        with open(config_module.__file__) as f:
            source = f.read()
        self.assertNotIn('GomokuNNet', source)
        self.assertNotIn('GomokuGame', source)


if __name__ == '__main__':
    unittest.main()
