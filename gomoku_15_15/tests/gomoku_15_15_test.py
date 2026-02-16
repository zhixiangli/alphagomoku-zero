#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy
import pytest

from alphazero.game import Game
from gomoku_15_15.game import GomokuGame, ChessType


@pytest.fixture
def gomoku15_game(make_args):
    return GomokuGame(make_args(rows=15, columns=15, n_in_row=5))


@pytest.mark.unit
def test_initial_state(gomoku15_game):
    board, player = gomoku15_game.get_initial_state()
    assert board == ""
    assert player == ChessType.BLACK


@pytest.mark.unit
def test_available_actions_empty_board(gomoku15_game):
    actions = gomoku15_game.available_actions("")
    assert len(actions) == 225
    assert actions == list(range(225))


@pytest.mark.unit
@pytest.mark.parametrize(
    "action,expected",
    [(7 * 15 + 7, "B[77]"), (10 * 15 + 14, "B[ae]"), (14 * 15 + 14, "B[ee]")],
    ids=["center", "hex_coords", "corner"],
)
def test_next_state_encodes_coordinates(gomoku15_game, action, expected):
    board, player = gomoku15_game.get_initial_state()
    next_board, next_player = gomoku15_game.next_state(board, action, player)
    assert next_board == expected
    assert next_player == ChessType.WHITE


@pytest.mark.unit
def test_hex_action_and_dec_action_support_high_coords(gomoku15_game):
    assert gomoku15_game.hex_action(161) == "[ab]"
    assert gomoku15_game.hex_action(224) == "[ee]"
    assert gomoku15_game.dec_action("B[ae]") == (10, 14)
    assert gomoku15_game.dec_action("W[ee]") == (14, 14)


@pytest.mark.unit
@pytest.mark.parametrize(
    "moves,last_action,player",
    [
        ([(ChessType.BLACK, 7, c) for c in range(3, 8)], 7 * 15 + 7, ChessType.BLACK),
        (
            [(ChessType.WHITE, r, 10) for r in range(5, 10)],
            9 * 15 + 10,
            ChessType.WHITE,
        ),
        (
            [(ChessType.BLACK, 10 + i, 10 + i) for i in range(5)],
            14 * 15 + 14,
            ChessType.BLACK,
        ),
        (
            [(ChessType.BLACK, i, 14 - i) for i in range(5)],
            4 * 15 + 10,
            ChessType.BLACK,
        ),
    ],
    ids=["horizontal", "vertical", "diagonal", "anti_diagonal"],
)
def test_terminal_wins_for_all_line_directions(gomoku15_game, build_sgf, moves, last_action, player):
    board = build_sgf(moves)
    assert gomoku15_game.is_terminal_state(board, last_action, player) == player


@pytest.mark.unit
def test_four_in_a_row_is_not_terminal_when_five_required(gomoku15_game, build_sgf):
    board = build_sgf([(ChessType.BLACK, 0, c) for c in range(4)])
    assert gomoku15_game.is_terminal_state(board, 3, ChessType.BLACK) is None


@pytest.mark.unit
def test_canonical_form_shape_and_channels(gomoku15_game):
    canonical = gomoku15_game.get_canonical_form("B[77];W[78]", ChessType.BLACK)
    assert canonical.shape == (15, 15, 2)
    assert canonical[7, 7, 0] == 1
    assert canonical[7, 8, 1] == 1
    assert canonical[7, 7, 1] == 0
    assert canonical[7, 8, 0] == 0


@pytest.mark.unit
def test_canonical_form_supports_hex_coordinates(gomoku15_game):
    canonical = gomoku15_game.get_canonical_form("B[ae];W[ea]", ChessType.BLACK)
    assert canonical[10, 14, 0] == 1
    assert canonical[14, 10, 1] == 1


@pytest.mark.unit
def test_available_actions_exclude_played_moves(gomoku15_game):
    actions = gomoku15_game.available_actions("B[77];W[78]")
    assert len(actions) == 223
    assert 7 * 15 + 7 not in actions
    assert 7 * 15 + 8 not in actions


@pytest.mark.unit
def test_augment_samples_preserves_shape_probability_and_value(gomoku15_game):
    board = numpy.zeros((15, 15, 2))
    board[0, 0, 0] = 1
    policy = numpy.zeros(225)
    policy[0] = 1.0

    augmented = gomoku15_game.augment_samples([(board, policy, 1.0)])
    assert len(augmented) == 8

    for aug_board, aug_policy, aug_value in augmented:
        assert aug_board.shape == (15, 15, 2)
        assert len(aug_policy) == 225
        assert numpy.sum(aug_policy) == pytest.approx(1.0)
        assert aug_value == 1.0


@pytest.mark.unit
def test_structure_sgf_and_to_board(gomoku15_game):
    sgf = "B[ae];W[ea];B[00]"
    parsed = gomoku15_game.structure_sgf(sgf)
    assert parsed == [("B", (10, 14)), ("W", (14, 10)), ("B", (0, 0))]

    board = gomoku15_game.to_board("B[00];W[ee]")
    assert board[0, 0] == ChessType.BLACK
    assert board[14, 14] == ChessType.WHITE
    assert board[7, 7] == ChessType.EMPTY


@pytest.mark.unit
def test_reward_contract(gomoku15_game):
    assert gomoku15_game.compute_reward(ChessType.BLACK, ChessType.BLACK) == 1
    assert gomoku15_game.compute_reward(ChessType.BLACK, ChessType.WHITE) == -1
    assert gomoku15_game.compute_reward(Game.DRAW, ChessType.BLACK) == 0


@pytest.mark.integration
def test_config_defaults_and_paths():
    from gomoku_15_15.config import GomokuConfig

    config = GomokuConfig()
    assert config.rows == 15
    assert config.columns == 15
    assert config.n_in_row == 5
    assert config.action_space_size == 225
    assert config.simulation_num == 900
    assert config.temp_step == 8
    assert config.dirichlet_alpha == 0.05
    assert config.dirichlet_epsilon == 0.10
    assert config.batch_size == 2048
    assert "gomoku_15_15" in config.save_checkpoint_path
    assert "gomoku_15_15" in config.sample_pool_file


@pytest.mark.integration
def test_trainer_and_module_wiring_importable():
    from gomoku_15_15 import configure_module
    from gomoku_15_15.trainer import main

    assert callable(main)
    assert callable(configure_module)
