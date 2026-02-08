#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import unittest

import numpy
from dotdict import dotdict

from alphazero.env import Env
from alphazero.mcts import MCTS
from alphazero.nnet import NNet
from alphazero.rl import RL


class ChessType:
    BLACK = 'B'
    WHITE = 'W'
    EMPTY = '.'


class MockEnv(Env):
    """
    1 * 3 board, 2 continuous color will be win.
    """
    rows = 1
    columns = 3

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
        return ChessType.EMPTY * MockEnv.columns, ChessType.BLACK

    def available_actions(self, board):
        return [i for i in range(len(board)) if board[i] == ChessType.EMPTY]

    def log_status(self, board, counts, actions):
        pass


class MockNNet(NNet):
    def predict(self, data):
        return numpy.array([1] * MockEnv.columns), 0

    def load_weights(self, filename):
        pass


class TestAlphaZero(unittest.TestCase):

    def setUp(self):
        self.env = MockEnv()
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
        self.mcts = MCTS(self.nnet, self.env, self.args)
        self.rl = RL(self.nnet, self.env, self.args)

    def test_mcts(self):
        board, player = self.env.get_initial_state()
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

    def test_select_chooses_best_ucb(self):
        board = '...'
        self.mcts.prior_probability[board] = numpy.array([0.2, 0.5, 0.3])
        self.mcts.visit_count[board] = numpy.array([5.0, 3.0, 2.0])
        self.mcts.mean_action_value[board] = numpy.array([0.1, 0.8, 0.3])
        self.mcts.total_visit_count[board] = 10
        self.mcts.available_actions[board] = [0, 1, 2]

        index = self.mcts._select(board)
        # Verify the selected index has the highest UCB value
        ucb = self.args.c_puct * self.mcts.prior_probability[board] * numpy.sqrt(
            self.mcts.total_visit_count[board]) / (
                    1.0 + self.mcts.visit_count[board]) + self.mcts.mean_action_value[board]
        self.assertEqual(index, numpy.argmax(ucb))

    def test_backup_updates_stats(self):
        board = '...'
        self.mcts.visit_count[board] = numpy.array([2.0, 0.0])
        self.mcts.mean_action_value[board] = numpy.array([0.5, 0.0])
        self.mcts.total_visit_count[board] = 2

        self.mcts._backup(board, 0, 1.0)
        self.assertAlmostEqual(self.mcts.mean_action_value[board][0], (0.5 * 2 + 1.0) / 3.0)
        self.assertEqual(self.mcts.visit_count[board][0], 3.0)
        self.assertEqual(self.mcts.total_visit_count[board], 3)

        self.mcts._backup(board, 1, -0.5)
        self.assertAlmostEqual(self.mcts.mean_action_value[board][1], -0.5)
        self.assertEqual(self.mcts.visit_count[board][1], 1.0)
        self.assertEqual(self.mcts.total_visit_count[board], 4)

    def test_expand_initializes_node(self):
        board, player = self.env.get_initial_state()
        value = self.mcts._expand(board, player)
        self.assertEqual(value, 0)
        self.assertEqual(len(self.mcts.available_actions[board]), 3)
        self.assertAlmostEqual(numpy.sum(self.mcts.prior_probability[board]), 1.0)
        self.assertEqual(self.mcts.total_visit_count[board], 0)
        numpy.testing.assert_array_equal(self.mcts.visit_count[board], numpy.zeros(3))
        numpy.testing.assert_array_equal(self.mcts.mean_action_value[board], numpy.zeros(3))

    def test_expand_handles_zero_proba(self):
        """Test that _expand handles neural network outputting all zeros."""
        class ZeroNNet(NNet):
            def predict(self, data):
                return numpy.array([0, 0, 0]), 0.5
        mcts = MCTS(ZeroNNet(), self.env, self.args)
        board, player = self.env.get_initial_state()
        value = mcts._expand(board, player)
        self.assertEqual(value, 0.5)
        # Should fall back to uniform distribution
        numpy.testing.assert_array_almost_equal(
            mcts.prior_probability[board], numpy.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))

    def test_total_visit_count_consistent(self):
        """Verify total_visit_count equals sum of visit_count after simulation."""
        board, player = self.env.get_initial_state()
        self.mcts.simulate(board, player)
        for b in self.mcts.total_visit_count:
            self.assertEqual(self.mcts.total_visit_count[b], numpy.sum(self.mcts.visit_count[b]))

    def test_play_against_itself(self):
        samples = self.rl.play_against_itself()
        print("play_against_itself", samples)


if __name__ == '__main__':
    unittest.main()
