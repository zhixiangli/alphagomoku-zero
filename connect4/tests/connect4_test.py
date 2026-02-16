#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy
import pytest
from unittest.mock import patch

from alphazero.game import Game
from alphazero.module import AlphaZeroModule
from alphazero.nnet import AlphaZeroNNet
from connect4 import configure_module
from connect4.config import Connect4Config
from connect4.game import Connect4Game, ChessType


@pytest.fixture
def connect4_game(make_args):
    return Connect4Game(make_args(rows=6, columns=7, n_in_row=4))


@pytest.mark.unit
def test_drop_gravity_in_same_column(connect4_game):
    board, player = connect4_game.get_initial_state()
    board, player = connect4_game.next_state(board, 0, player)
    assert board == "B[50]", "First token in a column should land on bottom row"

    board, _ = connect4_game.next_state(board, 0, player)
    assert board == "B[50];W[40]", "Second token in same column should stack upward"


@pytest.mark.unit
def test_available_actions_return_top_slot_per_column(connect4_game):
    assert connect4_game.available_actions("") == [35, 36, 37, 38, 39, 40, 41]


@pytest.mark.unit
def test_available_actions_update_after_partial_fill(connect4_game):
    actions = connect4_game.available_actions("B[50];W[40];B[30]")
    assert actions[0] == 14, "Column 0 should now expose row 2 as the legal action"


@pytest.mark.unit
@pytest.mark.parametrize(
    "board,action",
    [
        ("B[50];B[40];B[30];B[20]", 14),
        ("B[50];B[51];B[52];B[53]", 38),
        ("B[50];B[41];B[32];B[23]", 17),
    ],
    ids=["vertical", "horizontal", "diagonal"],
)
def test_terminal_win_patterns_are_detected(connect4_game, board, action):
    winner = connect4_game.is_terminal_state(board, action, ChessType.BLACK)
    assert winner == ChessType.BLACK


@pytest.mark.unit
def test_draw_when_board_is_full_without_winner(make_args):
    game = Connect4Game(make_args(rows=2, columns=2, n_in_row=3))
    board = "B[10];W[11];B[00];W[01]"
    assert game.is_terminal_state(board, 1, ChessType.WHITE) == Game.DRAW


@pytest.mark.unit
def test_canonical_form_swaps_current_and_opponent_channels(connect4_game):
    canonical = connect4_game.get_canonical_form("B[50];W[51];B[41]", ChessType.WHITE)
    assert canonical[5, 1, 0] == 1
    assert canonical[5, 0, 1] == 1
    assert canonical[4, 1, 1] == 1


@pytest.mark.unit
def test_augment_samples_mirrors_policy_probabilities(connect4_game):
    board = numpy.zeros((6, 7, 2))
    board[5, 0, 0] = 1
    policy = numpy.zeros(42)
    policy[35] = 1

    augmented = connect4_game.augment_samples([(board, policy, 1)])
    assert len(augmented) == 2
    _, flipped_policy, _ = augmented[1]
    assert flipped_policy[41] == 1, "Mirrored move should map from col=0 to col=6"


@pytest.mark.integration
def test_configure_module_registers_connect4_nnet():
    module = AlphaZeroModule()
    configure_module(module)
    assert module.resolve_nnet_class(Connect4Game) is AlphaZeroNNet


@pytest.mark.integration
def test_trainer_main_wires_cli_config():
    with patch("connect4.trainer.run_training") as mock_run, patch(
        "connect4.trainer.setup_logging"
    ), patch("sys.argv", ["trainer", "-rows", "6", "-columns", "7"]):
        from connect4.trainer import main

        main()

    assert mock_run.call_count == 1
    args, _ = mock_run.call_args
    config = args[2]
    assert isinstance(config, Connect4Config)
    assert config.rows == 6
    assert config.columns == 7
    assert config.batch_size == 256
