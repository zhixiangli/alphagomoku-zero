#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from gomoku.nnet import GomokuNNet


class TicTacToeNNet(GomokuNNet):
    """Tic-Tac-Toe-specific neural network adapter.

    Inherits all functionality from GomokuNNet since the feature
    extraction and prediction logic is identical for grid-based
    stone-placing games.
    """
    pass
