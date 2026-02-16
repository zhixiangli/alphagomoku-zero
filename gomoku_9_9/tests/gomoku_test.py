#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import logging

import numpy
import pytest

from alphazero.game import Game
from alphazero.nnet import AlphaZeroNNet
from gomoku_9_9.game import ChessType, GomokuGame


@pytest.fixture
def gomoku9_args(make_args):
    return make_args(rows=3, columns=3, n_in_row=2)


@pytest.fixture
def gomoku9_game(gomoku9_args):
    return GomokuGame(gomoku9_args)


@pytest.fixture
def gomoku9_nnet(gomoku9_game, gomoku9_args):
    return AlphaZeroNNet(gomoku9_game, gomoku9_args)


@pytest.mark.unit
def test_next_player(gomoku9_game):
    assert gomoku9_game.next_player(ChessType.BLACK) == ChessType.WHITE
    assert gomoku9_game.next_player(ChessType.WHITE) == ChessType.BLACK


@pytest.mark.unit
def test_next_state_appends_moves_in_sgf_order(gomoku9_game):
    board, player = gomoku9_game.get_initial_state()
    board, player = gomoku9_game.next_state(board, 6, player)
    assert board == "B[20]"
    assert player == ChessType.WHITE

    board, player = gomoku9_game.next_state(board, 0, player)
    assert board == "B[20];W[00]"
    assert player == ChessType.BLACK


@pytest.mark.unit
def test_terminal_contract_for_win_draw_and_non_terminal(gomoku9_game):
    full_black = ";".join(
        ChessType.BLACK + gomoku9_game.hex_action(i) for i in range(gomoku9_game.rows * gomoku9_game.columns)
    )
    assert gomoku9_game.is_terminal_state(full_black, 0, ChessType.BLACK) == ChessType.BLACK

    full_white = ";".join(
        ChessType.WHITE + gomoku9_game.hex_action(i) for i in range(gomoku9_game.rows * gomoku9_game.columns)
    )
    assert gomoku9_game.is_terminal_state(full_white, 0, ChessType.BLACK) == Game.DRAW

    assert gomoku9_game.is_terminal_state("B[03];B[10]", 3, ChessType.BLACK) is None


@pytest.mark.unit
def test_available_actions_returns_unoccupied_cells(gomoku9_game):
    board = ";".join(
        ChessType.BLACK + gomoku9_game.hex_action(i) for i in range(0, gomoku9_game.rows * gomoku9_game.columns, 2)
    )
    expected = [i for i in range(1, gomoku9_game.rows * gomoku9_game.columns, 2)]
    assert gomoku9_game.available_actions(board) == expected


@pytest.mark.unit
def test_canonical_form_respects_current_player_perspective(gomoku9_game):
    canonical_white = gomoku9_game.get_canonical_form("B[20];W[21];B[11]", ChessType.WHITE)
    numpy.testing.assert_array_equal(
        canonical_white,
        numpy.array(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 1], [0, 0]],
                [[0, 1], [1, 0], [0, 0]],
            ]
        ),
    )

    canonical_black = gomoku9_game.get_canonical_form("B[20];W[21];B[11]", ChessType.BLACK)
    numpy.testing.assert_array_equal(
        canonical_black,
        numpy.array(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [1, 0], [0, 0]],
                [[1, 0], [0, 1], [0, 0]],
            ]
        ),
    )


@pytest.mark.unit
def test_empty_canonical_form_is_all_zeros(gomoku9_game):
    numpy.testing.assert_array_equal(
        gomoku9_game.get_canonical_form("", ChessType.BLACK),
        numpy.zeros((3, 3, 2)),
    )


@pytest.mark.unit
def test_helper_methods_convert_actions_and_sgf(gomoku9_game):
    assert gomoku9_game.hex_action(4) == "[11]"
    assert gomoku9_game.dec_action("B[12]") == (1, 2)
    assert gomoku9_game.structure_sgf("B[00];W[11];B[22]") == [
        ("B", (0, 0)),
        ("W", (1, 1)),
        ("B", (2, 2)),
    ]


@pytest.mark.unit
def test_to_board_maps_stones_and_preserves_empty_cells(gomoku9_game):
    board = gomoku9_game.to_board("B[00];W[11]")
    assert board[0, 0] == ChessType.BLACK
    assert board[1, 1] == ChessType.WHITE
    assert board[0, 1] == ChessType.EMPTY
    assert board.shape == (3, 3)


@pytest.mark.slow
def test_checkpoint_round_trip(gomoku9_nnet, tmp_path):
    prefix = str(tmp_path / "gomoku_ckpt")
    gomoku9_nnet.save_checkpoint(prefix)

    files = glob.glob(prefix + "*.pt")
    assert len(files) == 1, "Expected exactly one checkpoint file for deterministic loading"

    gomoku9_nnet.load_checkpoint(prefix)


@pytest.mark.slow
def test_checkpoint_load_missing_is_noop(gomoku9_nnet, tmp_path):
    gomoku9_nnet.load_checkpoint(str(tmp_path / "missing_prefix"))


@pytest.mark.slow
def test_train_logs_epoch_progress(gomoku9_game, make_args, caplog):
    args = make_args(rows=3, columns=3, n_in_row=2, batch_size=4, epochs=2)
    nnet = AlphaZeroNNet(gomoku9_game, args)

    data = [
        (numpy.zeros((3, 3, 2)), numpy.ones(9) / 9, 1.0)
        for _ in range(8)
    ]

    with caplog.at_level(logging.INFO):
        nnet.train(data)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "training start: 8 samples, 2 epochs, batch_size=4" in messages
    assert "epoch 1/2" in messages
    assert "epoch 2/2" in messages
    assert "policy_loss:" in messages
    assert "value_loss:" in messages
    assert "training complete:" in messages


@pytest.mark.unit
def test_value_head_contract(gomoku9_nnet):
    model = gomoku9_nnet.model
    assert model.value_fc1.out_features == 256
    assert model.value_fc2.out_features == 1
