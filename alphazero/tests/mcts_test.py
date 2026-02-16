#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Behavioral tests for Monte Carlo Tree Search contracts."""

import numpy
import pytest
from dotdict import dotdict

from alphazero.game import Game
from alphazero.mcts import MCTS
from alphazero.nnet import NNet


class _ChessType:
    BLACK = "B"
    WHITE = "W"
    EMPTY = "."


class SimpleBoardGame(Game):
    """Minimal 1x3 board game where adjacent same-color stones win."""

    rows = 1
    columns = 3

    def next_player(self, player):
        return _ChessType.BLACK if player == _ChessType.WHITE else _ChessType.WHITE

    def next_state(self, board, action, player):
        stones = list(board)
        stones[action] = player
        return "".join(stones), self.next_player(player)

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
        return [i for i, stone in enumerate(board) if stone == _ChessType.EMPTY]

    def log_status(self, board, counts, actions):
        return None

    def get_canonical_form(self, board, player):
        if player == _ChessType.BLACK:
            return board
        return "".join(
            self.next_player(c) if c in (_ChessType.BLACK, _ChessType.WHITE) else c
            for c in board
        )


class UniformNNet(NNet):
    def predict(self, board):
        return numpy.ones(3), 0


@pytest.fixture
def tiny_game():
    return SimpleBoardGame()


@pytest.fixture
def uniform_nnet():
    return UniformNNet()


@pytest.fixture
def mcts_args():
    return dotdict({"simulation_num": 100, "c_puct": 5})


@pytest.mark.unit
def test_simulate_returns_aligned_actions_and_counts(tiny_game, uniform_nnet, mcts_args):
    actions, counts = MCTS(uniform_nnet, tiny_game, mcts_args).simulate(*tiny_game.get_initial_state())
    assert len(actions) == len(counts)
    assert len(actions) > 0


@pytest.mark.unit
def test_simulate_distributes_all_visits_across_children(tiny_game, uniform_nnet, mcts_args):
    _, counts = MCTS(uniform_nnet, tiny_game, mcts_args).simulate(*tiny_game.get_initial_state())
    assert int(numpy.sum(counts)) == mcts_args.simulation_num - 1


@pytest.mark.unit
def test_simulate_prefers_center_and_keeps_symmetric_edges(tiny_game, uniform_nnet, mcts_args):
    actions, counts = MCTS(uniform_nnet, tiny_game, mcts_args).simulate(*tiny_game.get_initial_state())
    action_list = list(actions)

    center_index = action_list.index(1)
    assert int(numpy.argmax(counts)) == center_index, "Center move should be most visited on symmetric board"

    left_index = action_list.index(0)
    right_index = action_list.index(2)
    assert int(counts[left_index]) == int(counts[right_index]), "Symmetric edge moves should receive equal visits"


@pytest.mark.unit
def test_simulate_caches_expanded_states_and_terminal_outcomes(tiny_game, uniform_nnet):
    mcts = MCTS(uniform_nnet, tiny_game, dotdict({"simulation_num": 50, "c_puct": 5}))
    board, player = tiny_game.get_initial_state()
    mcts.simulate(board, player)

    assert board in mcts.prior_probability
    assert board in mcts.visit_count
    assert board in mcts.mean_action_value
    assert sum(1 for outcome in mcts.terminal_state.values() if outcome is not None) > 0


@pytest.mark.unit
def test_simulate_applies_dirichlet_noise_to_root_when_enabled(tiny_game, uniform_nnet):
    args = dotdict(
        {
            "simulation_num": 1,
            "c_puct": 5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
        }
    )
    mcts = MCTS(uniform_nnet, tiny_game, args)
    board, player = tiny_game.get_initial_state()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("numpy.random.dirichlet", lambda *_: numpy.array([0.7, 0.2, 0.1]))
        mcts.simulate(board, player, add_root_noise=True)

    expected = (1.0 - args.dirichlet_epsilon) * (numpy.ones(3) / 3.0) + (
        args.dirichlet_epsilon * numpy.array([0.7, 0.2, 0.1])
    )
    numpy.testing.assert_allclose(mcts.prior_probability[board], expected)


@pytest.mark.unit
def test_search_expands_beyond_root_with_repeated_calls(tiny_game, uniform_nnet):
    mcts = MCTS(uniform_nnet, tiny_game, dotdict({"simulation_num": 10, "c_puct": 5}))
    board, player = tiny_game.get_initial_state()

    mcts.search(board, player)
    assert board in mcts.prior_probability

    for _ in range(20):
        mcts.search(board, player)

    assert len(mcts.prior_probability) > 1


@pytest.mark.unit
def test_simulate_handles_single_forced_winning_move(tiny_game, uniform_nnet):
    mcts = MCTS(uniform_nnet, tiny_game, dotdict({"simulation_num": 200, "c_puct": 5}))

    actions, counts = mcts.simulate("B.W", _ChessType.BLACK)
    assert list(actions) == [1]
    assert int(counts[0]) == 199
