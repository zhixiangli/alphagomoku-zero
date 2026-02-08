#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from alphazero.mcts import MCTS
from alphazero.nnet import NNet
from alphazero.rl import RL


class ChessType:
    BLACK = 'B'
    WHITE = 'W'
    EMPTY = '.'


class MockGame(Game):
    """
    1 * 3 board, 2 continuous color will be win.
    """
    rows = 1
    columns = 3

    @property
    def action_space_size(self):
        return MockGame.rows * MockGame.columns

    def next_player(self, player):
        assert player != ChessType.EMPTY
        return ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE

    def next_state(self, board, action, player):
        assert 0 <= action < len(board)
        assert board[action] == ChessType.EMPTY
        assert player != ChessType.EMPTY
        tmp = list(board)
        tmp[action] = player
        return ''.join(tmp), self.next_player(player)

    def is_terminal_state(self, board, action, player):
        assert player != ChessType.EMPTY
        assert board[action] == player
        if (action > 0 and board[action - 1] == board[action]) or (action == 0 and board[action + 1] == board[action]):
            return player
        if all(ch != ChessType.EMPTY for ch in board):
            return ChessType.EMPTY
        return None

    def get_initial_state(self):
        return ChessType.EMPTY * MockGame.columns, ChessType.BLACK

    def available_actions(self, board):
        return [i for i in range(len(board)) if board[i] == ChessType.EMPTY]

    def log_status(self, board, counts, actions):
        pass

    def compute_reward(self, winner, player):
        if winner == ChessType.EMPTY:
            return 0
        return 1 if player == winner else -1


class MockNNet(NNet):
    def predict(self, data):
        return numpy.array([1] * MockGame.columns), 0

    def load_weights(self, filename):
        pass


class TestAlphaZero(unittest.TestCase):

    def setUp(self):
        self.game = MockGame()
        self.nnet = MockNNet()
        self.args = dotdict({
            'simulation_num': 100,
            'c_puct': 5,
            'save_weights_path': '',
            'rows': 1,
            'columns': 3,
            'max_sample_pool_size': 100000,
            'sample_pool_file': '',
            'temp_step': 5,
        })
        self.mcts = MCTS(self.nnet, self.game, self.args)
        self.rl = RL(self.nnet, self.game, self.args)

    def test_mcts(self):
        board, player = self.game.get_initial_state()
        self.mcts.simulate(board, player)
        print("visit_count", self.mcts.visit_count)
        print("mean_action_value", self.mcts.mean_action_value)
        print("prior_probability", self.mcts.prior_probability)
        print("terminal_state", self.mcts.terminal_state)
        print("total_visit_count", self.mcts.total_visit_count)
        print("available_actions", self.mcts.available_actions)

        counts = self.mcts.visit_count[board]
        self.assertTrue(counts[0] < counts[1])
        self.assertTrue(counts[-1] < counts[-2])

    def test_play_against_itself(self):
        samples = self.rl.play_against_itself()
        print("play_against_itself", samples)


if __name__ == '__main__':
    unittest.main()
