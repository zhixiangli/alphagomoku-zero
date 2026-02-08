#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import os
import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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

    def train(self, data):
        pass

    def save_weights(self, filename):
        pass

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

    def test_play_against_itself(self):
        samples = self.rl.play_against_itself()
        print("play_against_itself", samples)

    def test_play_against_itself_returns_valid_samples(self):
        samples = self.rl.play_against_itself()
        self.assertIsNotNone(samples)
        self.assertGreater(len(samples), 0)
        for board, player, policy, value in samples:
            self.assertIn(player, [ChessType.BLACK, ChessType.WHITE])
            self.assertEqual(len(policy), self.args.rows * self.args.columns)
            self.assertIn(value, [-1, 0, 1])

    def test_play_against_itself_draw_values_are_zero(self):
        samples = self.rl.play_against_itself()
        for board, player, policy, value in samples:
            if value == 0:
                # Draw means all values should be 0
                all_values = [v for _, _, _, v in samples]
                if 0 in all_values:
                    self.assertTrue(all(v == 0 for v in all_values))
                break

    def test_create_mcts_returns_mcts_instance(self):
        mcts = self.rl.create_mcts()
        self.assertIsInstance(mcts, MCTS)

    def test_create_mcts_can_be_overridden(self):
        mock_mcts = MagicMock()
        mock_mcts.simulate.return_value = (
            numpy.array([0, 1, 2]),
            numpy.array([10.0, 20.0, 10.0])
        )

        class CustomRL(RL):
            def create_mcts(self):
                return mock_mcts

        rl = CustomRL(self.nnet, self.env, self.args)
        mcts = rl.create_mcts()
        self.assertIs(mcts, mock_mcts)

    def test_start_with_num_iterations(self):
        self.args.batch_size = 1000000
        self.args.persist_interval = 10
        self.rl.start(num_iterations=2)
        self.assertGreater(len(self.rl.sample_pool), 0)

    def test_start_trains_when_enough_samples(self):
        self.args.batch_size = 1
        self.args.persist_interval = 100
        self.nnet.train = MagicMock()
        self.rl.start(num_iterations=2)
        self.assertTrue(self.nnet.train.called)

    def test_start_persists_at_interval(self):
        self.args.batch_size = 1000000
        self.args.persist_interval = 1
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            self.args.sample_pool_file = f.name
        try:
            self.rl.persist_sample_pool = MagicMock()
            self.rl.nnet.save_weights = MagicMock()
            self.args.batch_size = 1
            self.rl.start(num_iterations=1)
            self.assertTrue(self.rl.nnet.save_weights.called)
        finally:
            if os.path.exists(self.args.sample_pool_file):
                os.unlink(self.args.sample_pool_file)

    def test_persist_and_read_sample_pool(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            self.args.sample_pool_file = f.name
        try:
            test_data = [(1, 2, 3), (4, 5, 6)]
            self.rl.persist_sample_pool(test_data)
            loaded = self.rl.read_sample_pool()
            self.assertEqual(loaded, test_data)
        finally:
            os.unlink(self.args.sample_pool_file)

    def test_read_sample_pool_returns_none_for_missing_file(self):
        self.args.sample_pool_file = '/tmp/nonexistent_test_file.pkl'
        rl = RL(self.nnet, self.env, self.args)
        self.assertEqual(len(rl.sample_pool), 0)

    def test_proba_normalization(self):
        """Verify that proba normalization prevents floating-point issues."""
        pi = numpy.array([0.3333333, 0.3333333, 0.3333334])
        dirichlet = numpy.array([0.5, 0.3, 0.2])
        proba = 0.75 * pi + 0.25 * dirichlet
        proba /= proba.sum()
        self.assertAlmostEqual(proba.sum(), 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
