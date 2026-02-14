#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Play Gomoku (15x15) against a trained AlphaZero model via stdio."""

import re

from alphazero.stdio_play import run_stdio_game as _run_stdio_game
from gomoku_15_15.config import GomokuConfig
from gomoku_15_15.game import ChessType, GomokuGame

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


def run_stdio_game(config_class, game_class, chess_type, title):
    max_col = _column_label(config_class.columns - 1)
    _run_stdio_game(
        config_class=config_class,
        game_class=game_class,
        chess_type=chess_type,
        title=title,
        description=f"Play {config_class.rows}x{config_class.columns} Gomoku against a trained model",
        parse_move=_parse_move,
        format_action=_format_action,
        print_board=_print_board,
        help_message=(
            "Play with A1 or 'A 1' (example: E5).\n"
            f"Try A1..{max_col}{config_class.rows} (or A 1). "
            f"Columns A-{max_col}, rows 1-{config_class.rows}."
        ),
        invalid_move_message="That move doesn't work. Pick an empty spot on the board.",
    )


def main():
    run_stdio_game(GomokuConfig, GomokuGame, ChessType, title="Gomoku 15x15")


if __name__ == "__main__":
    main()
