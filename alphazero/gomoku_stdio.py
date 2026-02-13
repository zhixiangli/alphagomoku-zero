#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Shared stdio runner for human-vs-AI Gomoku play."""

import argparse
import logging
import re

import numpy

from alphazero.mcts import MCTS
from alphazero.nnet import AlphaZeroNNet

_COLUMN_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _column_label(index):
    return _COLUMN_LABELS[index]


def _format_action(game, action):
    row, col = divmod(action, game.args.columns)
    return f"{_column_label(col)}{row + 1}"


def _parse_move(text, rows, columns):
    normalized = text.strip().upper()

    if re.fullmatch(r"[A-Z]\d+", normalized):
        col = ord(normalized[0]) - ord("A")
        row = int(normalized[1:]) - 1
        if 0 <= row < rows and 0 <= col < columns:
            return row * columns + col

    parts = normalized.split()
    if (
        len(parts) == 2
        and len(parts[0]) == 1
        and parts[0].isalpha()
        and parts[1].isdigit()
    ):
        col = ord(parts[0]) - ord("A")
        row = int(parts[1]) - 1
        if 0 <= row < rows and 0 <= col < columns:
            return row * columns + col

    return None


def _print_board(game, board):
    matrix = game.to_board(board)
    col_header = "   " + " ".join(_column_label(i) for i in range(game.args.columns))
    print("\n" + col_header)
    for i in range(game.args.rows):
        row_label = f"{i + 1:>2}"
        print(f"{row_label} " + " ".join(matrix[i]))
    print()


def _pick_ai_action(mcts, board, player):
    actions, counts = mcts.simulate(board, player)
    if len(actions) == 0:
        return None
    best = numpy.max(counts)
    best_actions = actions[counts == best]
    return int(numpy.random.choice(best_actions))


def build_args_parser(description, default_checkpoint_path, chess_type):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--human-color",
        choices=(chess_type.BLACK, chess_type.WHITE),
        default=chess_type.BLACK,
        help=f"Choose your side: {chess_type.BLACK} (first) or {chess_type.WHITE} (second).",
    )
    parser.add_argument(
        "--simulation-num",
        type=int,
        default=None,
        help="Override MCTS simulations per AI move (defaults to config value).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=default_checkpoint_path,
        help="Checkpoint prefix passed to load_checkpoint.",
    )
    return parser


def run_stdio_game(config_class, game_class, chess_type, title):
    parser = build_args_parser(
        description=f"Play {config_class.rows}x{config_class.columns} Gomoku against a trained model",
        default_checkpoint_path=config_class.save_checkpoint_path,
        chess_type=chess_type,
    )
    cli_args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = config_class()
    if cli_args.simulation_num is not None:
        config.simulation_num = cli_args.simulation_num

    game = game_class(config)
    nnet = AlphaZeroNNet(game, config)
    nnet.load_checkpoint(cli_args.checkpoint_path)
    mcts = MCTS(nnet, game, config)

    board, player = game.get_initial_state()
    human = cli_args.human_color
    ai = game.next_player(human)

    max_col = _column_label(config.columns - 1)
    print(f"Welcome to {title} 👋")
    print("Play with A1 or 'A 1' (example: E5).")
    print("Type 'help' for tips, 'quit' to leave.\n")

    while True:
        _print_board(game, board)

        if player == human:
            raw = input(f"Your move ({human}): ").strip()
            if raw.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if raw.lower() in {"help", "h", "?"}:
                print(
                    f"Try A1..{max_col}{config.rows} (or A 1). "
                    f"Columns A-{max_col}, rows 1-{config.rows}."
                )
                continue

            action = _parse_move(raw, config.rows, config.columns)
            available = game.available_actions(board)
            if action is None or action not in available:
                print("That move doesn't work. Pick an empty spot on the board.")
                continue

            board, next_player = game.next_state(board, action, player)
            winner = game.is_terminal_state(board, action, player)
            print(f"Nice move: {_format_action(game, action)}")
        else:
            print("AI is thinking…")
            action = _pick_ai_action(mcts, board, player)
            if action is None:
                print("No legal moves left.")
                break
            board, next_player = game.next_state(board, action, player)
            winner = game.is_terminal_state(board, action, player)
            print(f"AI ({ai}) plays {_format_action(game, action)}")

        if winner is not None:
            _print_board(game, board)
            if winner == human:
                print("You win! 🎉")
            elif winner == ai:
                print("AI wins — good game!")
            else:
                print("Draw game.")
            break

        player = next_player
