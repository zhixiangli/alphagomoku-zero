#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy

from alphazero.nnet import AlphaZeroNNet
from gomoku.game import ChessType


class GomokuNNet(AlphaZeroNNet):
    """Gomoku-specific neural network adapter.

    Extends AlphaZeroNNet with Gomoku-specific feature extraction.
    Expects canonical board input (current player always represented as BLACK).
    """

    def train(self, data):
        boards, policies, values = zip(*data)
        states = numpy.zeros((len(boards), 2, self.args.rows, self.args.columns))
        for i in range(len(boards)):
            states[i] = self.fit_transform(boards[i])
        policies = numpy.array(policies)
        values = numpy.array(values)
        self.model.fit(x=states, y=[policies, values], batch_size=self.args.batch_size, epochs=self.args.epochs)

    def predict(self, board):
        states = numpy.zeros((1, 2, self.args.rows, self.args.columns))
        states[0] = self.fit_transform(board)
        policy, value = self.model.predict(states)
        return policy[0], value[0]

    def fit_transform(self, board):
        """Extract features from a canonical board.

        Channel 0: current player's stones (BLACK in canonical form)
        Channel 1: opponent's stones (WHITE in canonical form)
        """
        feature = numpy.zeros((2, self.args.rows, self.args.columns))
        if board:
            for stone in board.split(self.game.semicolon):
                if stone:
                    (x, y) = self.game.dec_action(stone)
                    if stone[0] == ChessType.BLACK:
                        feature[0][x][y] = 1
                    elif stone[0] == ChessType.WHITE:
                        feature[1][x][y] = 1
        return feature
