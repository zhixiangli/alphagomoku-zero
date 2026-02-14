#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Generic stdio runner for human-vs-AI board games."""

import argparse
import logging

import numpy

from alphazero.mcts import MCTS
from alphazero.nnet import AlphaZeroNNet


def _resolve_direct_action(parsed, available_actions):
    if parsed is None or parsed not in available_actions:
        return None
    return parsed


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


def run_stdio_game(
    *,
    config_class,
    game_class,
    chess_type,
    title,
    description,
    parse_move,
    format_action,
    print_board,
    help_message,
    invalid_move_message,
    resolve_action=None,
):
    parser = build_args_parser(
        description=description,
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

    if resolve_action is None:
        resolve_action = _resolve_direct_action

    print(f"Welcome to {title} 👋")
    print(help_message)
    print("Type 'help' for tips, 'quit' to leave.\n")

    while True:
        print_board(game, board)

        if player == human:
            raw = input(f"Your move ({human}): ").strip()
            if raw.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if raw.lower() in {"help", "h", "?"}:
                print(help_message)
                continue

            parsed = parse_move(raw, config.rows, config.columns)
            available = game.available_actions(board)
            action = resolve_action(parsed, available)
            if action is None:
                print(invalid_move_message)
                continue

            board, next_player = game.next_state(board, action, player)
            winner = game.is_terminal_state(board, action, player)
            print(f"Nice move: {format_action(game, action)}")
        else:
            print("AI is thinking…")
            action = _pick_ai_action(mcts, board, player)
            if action is None:
                print("No legal moves left.")
                break
            board, next_player = game.next_state(board, action, player)
            winner = game.is_terminal_state(board, action, player)
            print(f"AI ({ai}) plays {format_action(game, action)}")

        if winner is not None:
            print_board(game, board)
            if winner == human:
                print("You win! 🎉")
            elif winner == ai:
                print("AI wins — good game!")
            else:
                print("Draw game.")
            break

        player = next_player
