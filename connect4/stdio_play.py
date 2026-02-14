#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Play Connect4 against a trained AlphaZero model via stdio."""

from alphazero.stdio_play import run_stdio_game
from connect4.config import Connect4Config
from connect4.game import ChessType, Connect4Game


def _format_action(game, action):
    return str(action % game.args.columns + 1)


def _parse_move(text, rows, columns):
    raw = text.strip()
    if not raw.isdigit():
        return None
    col = int(raw) - 1
    if not 0 <= col < columns:
        return None
    return col


def _resolve_action(parsed, available_actions):
    if parsed is None:
        return None
    for action in available_actions:
        if action % Connect4Config.columns == parsed:
            return action
    return None


def _print_board(game, board):
    matrix = game.to_board(board)
    print("\n  " + " ".join(str(i + 1) for i in range(game.args.columns)))
    for i in range(game.args.rows):
        print(f"{i + 1:>2} " + " ".join(matrix[i]))
    print()


def main():
    last_col = Connect4Config.columns
    run_stdio_game(
        config_class=Connect4Config,
        game_class=Connect4Game,
        chess_type=ChessType,
        title="Connect4",
        description="Play Connect4 against a trained model",
        parse_move=_parse_move,
        format_action=_format_action,
        print_board=_print_board,
        help_message=f"Enter a column number from 1 to {last_col}.",
        invalid_move_message="That move doesn't work. Choose a non-full column.",
        resolve_action=_resolve_action,
    )


if __name__ == "__main__":
    main()
