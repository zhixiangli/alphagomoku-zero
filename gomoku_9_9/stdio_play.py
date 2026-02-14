#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Play Gomoku (9x9) against a trained AlphaZero model via stdio."""

from gomoku_15_15.stdio_play import run_stdio_game
from gomoku_9_9.config import GomokuConfig
from gomoku_9_9.game import ChessType, GomokuGame


def main():
    run_stdio_game(GomokuConfig, GomokuGame, ChessType, title="Gomoku 9x9")


if __name__ == "__main__":
    main()
