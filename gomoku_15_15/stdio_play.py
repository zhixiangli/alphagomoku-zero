#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Play Gomoku (15x15) against a trained AlphaZero model via stdio."""

from alphazero.gomoku_stdio import run_stdio_game
from gomoku_15_15.config import GomokuConfig
from gomoku_15_15.game import ChessType, GomokuGame


def main():
    run_stdio_game(GomokuConfig, GomokuGame, ChessType, title="Gomoku 15x15")


if __name__ == "__main__":
    main()
