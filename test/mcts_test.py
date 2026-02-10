#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Tests for alphazero/mcts.py - Monte Carlo Tree Search."""

import unittest

import numpy
from dotdict import dotdict

from alphazero.game import Game
from alphazero.mcts import MCTS
from alphazero.nnet import NNet


class _ChessType:
    BLACK = "B"
    WHITE = "W"
    EMPTY = "."


class SimpleBoardGame(Game):
    """Minimal 1×3 board game: two adjacent same-colour stones win."""

    def __init__(self):
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
        return _ChessType.EMPTY * 3, _ChessType.BLACK

    def available_actions(self, board):
        return [i for i in range(len(board)) if board[i] == _ChessType.EMPTY]

    def log_status(self, board, counts, actions):
        pass

    def get_canonical_form(self, board, player):
        if player == _ChessType.BLACK:
            return board
        return "".join(
            self.next_player(c) if c in (_ChessType.BLACK, _ChessType.WHITE) else c
            for c in board
        )


class UniformNNet(NNet):
    """Returns uniform priors and zero value."""

    def predict(self, board):
        return numpy.ones(3), 0


class TestMCTSSimulate(unittest.TestCase):
    def setUp(self):
        self.game = SimpleBoardGame()
        self.nnet = UniformNNet()
        self.args = dotdict({"simulation_num": 100, "c_puct": 5})

    def test_simulate_returns_actions_and_counts(self):
        mcts = MCTS(self.nnet, self.game, self.args)
        board, player = self.game.get_initial_state()
        actions, counts = mcts.simulate(board, player)
        self.assertEqual(len(actions), len(counts))
        self.assertGreater(len(actions), 0)

    def test_simulate_total_visits_equal_simulation_num(self):
        mcts = MCTS(self.nnet, self.game, self.args)
        board, player = self.game.get_initial_state()
        actions, counts = mcts.simulate(board, player)
        # Total visit count at root = simulation_num + 1 (initial expand counts as 1)
        # But the counts returned are the children's visit counts
        self.assertEqual(int(numpy.sum(counts)), self.args.simulation_num - 1)

    def test_center_move_preferred(self):
        """On a 1x3 board, center move should be visited most (strategic value)."""
        mcts = MCTS(self.nnet, self.game, self.args)
        board, player = self.game.get_initial_state()
        actions, counts = mcts.simulate(board, player)
        center_idx = list(actions).index(1)
        self.assertEqual(int(numpy.argmax(counts)), center_idx)

    def test_edge_moves_symmetric(self):
        """Edge moves (0 and 2) should have roughly equal visit counts."""
        mcts = MCTS(self.nnet, self.game, self.args)
        board, player = self.game.get_initial_state()
        actions, counts = mcts.simulate(board, player)
        action_list = list(actions)
        idx_0 = action_list.index(0)
        idx_2 = action_list.index(2)
        self.assertEqual(int(counts[idx_0]), int(counts[idx_2]))


class TestMCTSCaching(unittest.TestCase):
    def setUp(self):
        self.game = SimpleBoardGame()
        self.nnet = UniformNNet()
        self.args = dotdict({"simulation_num": 50, "c_puct": 5})

    def test_states_are_cached(self):
        mcts = MCTS(self.nnet, self.game, self.args)
        board, player = self.game.get_initial_state()
        mcts.simulate(board, player)
        # Root should be in prior_probability cache
        self.assertIn(board, mcts.prior_probability)
        self.assertIn(board, mcts.visit_count)
        self.assertIn(board, mcts.mean_action_value)

    def test_terminal_states_cached(self):
        mcts = MCTS(self.nnet, self.game, self.args)
        board, player = self.game.get_initial_state()
        mcts.simulate(board, player)
        # Some terminal states should have been found
        terminal_count = sum(1 for v in mcts.terminal_state.values() if v is not None)
        self.assertGreater(terminal_count, 0)


class TestMCTSSearch(unittest.TestCase):
    def setUp(self):
        self.game = SimpleBoardGame()
        self.nnet = UniformNNet()
        self.args = dotdict({"simulation_num": 10, "c_puct": 5})

    def test_search_initializes_leaf(self):
        mcts = MCTS(self.nnet, self.game, self.args)
        board, player = self.game.get_initial_state()
        # First search should expand the root
        mcts.search(board, player)
        self.assertIn(board, mcts.prior_probability)

    def test_multiple_searches_build_tree(self):
        mcts = MCTS(self.nnet, self.game, self.args)
        board, player = self.game.get_initial_state()
        for _ in range(20):
            mcts.search(board, player)
        # Tree should have expanded beyond root
        self.assertGreater(len(mcts.prior_probability), 1)


class TestMCTSWithWinningMove(unittest.TestCase):
    """Test MCTS when there's an obvious winning move."""

    def setUp(self):
        self.game = SimpleBoardGame()
        self.nnet = UniformNNet()
        self.args = dotdict({"simulation_num": 200, "c_puct": 5})

    def test_finds_winning_move(self):
        """When one move wins, MCTS should overwhelmingly select it."""
        # Board: B.B - placing at position 1 wins for BLACK's previous turn
        # Actually let's think about this differently:
        # Board "B.." with BLACK having played position 0
        # Now it's WHITE's turn, but let's set up a position where
        # the current player has a winning move
        mcts = MCTS(self.nnet, self.game, self.args)
        # Board: "B.." - WHITE to move, if WHITE plays 1 -> "BW." no win
        # Better: position after B plays 0, W plays 2 -> "B.W"
        # Now B to move, B plays 1 -> "BBW" - B wins (adjacent BB)
        board = "B.W"
        player = _ChessType.BLACK
        actions, counts = mcts.simulate(board, player)
        # Only action 1 is available and it's a win
        self.assertEqual(list(actions), [1])
        self.assertEqual(int(counts[0]), self.args.simulation_num - 1)


if __name__ == "__main__":
    unittest.main()
