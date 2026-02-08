#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy

from alphazero.nnet import AlphaZeroNNet
from gomoku.game import ChessType


class GomokuNNet(AlphaZeroNNet):
    """Gomoku-specific neural network adapter.

    Extends AlphaZeroNNet with Gomoku-specific feature extraction
    (board encoding and player-insensitive transforms).
    """

    def train(self, data):
        boards, players, policies, values = zip(*data)
        states = numpy.zeros((len(players), self.args.history_num * 2 + 1, self.args.rows, self.args.columns))
        for i in range(len(players)):
            states[i] = self.fit_transform(boards[i], players[i])
        policies = numpy.array(policies)
        values = numpy.array(values)
        self.model.fit(x=states, y=[policies, values], batch_size=self.args.batch_size, epochs=self.args.epochs)

    def predict(self, data):
        board, player = data
        states = numpy.zeros((1, self.args.history_num * 2 + 1, self.args.rows, self.args.columns))
        states[0] = self.fit_transform(board, player)
        policy, value = self.model.predict(states)
        return policy[0], value[0]

    def fit_transform(self, board, player):
        def transform(board, player):
            f = numpy.zeros((self.args.history_num, self.args.rows, self.args.columns))
            actions = [self.game.dec_action(stone) for stone in board.split(self.game.semicolon) if
                       stone and stone[0] == player]
            for i in range(self.args.history_num):
                for (x, y) in actions[:len(actions) - i]:
                    f[self.args.history_num - i - 1][x][y] = 1
            return f

        feature = numpy.zeros((self.args.history_num * 2 + 1, self.args.rows, self.args.columns))
        if player == ChessType.BLACK:
            feature[-1] = numpy.ones((self.args.rows, self.args.columns))
        new_board = self.player_insensitive_board(board, player)
        feature[:self.args.history_num] = transform(new_board, ChessType.BLACK)
        feature[self.args.history_num:self.args.history_num * 2] = transform(new_board, ChessType.WHITE)
        return feature

    def player_insensitive_board(self, board, player):
        assert player != ChessType.EMPTY
        if player == ChessType.BLACK:
            return board
        return "".join([c if c != ChessType.BLACK and c != ChessType.WHITE else self.game.next_player(c) for c in board])
