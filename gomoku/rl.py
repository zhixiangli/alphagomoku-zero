#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy

from alphazero.rl import RL


class GomokuRL(RL):

    def augment_samples(self, samples):
        augmented = []
        for sample in samples:
            board, player, policy, value = sample
            boards = self.augment_board(board)
            policies = self.augment_policy(policy)
            for i in range(len(boards)):
                augmented.append((boards[i], player, policies[i], value))
        return augmented

    def augment_board(self, board):
        colors, actions = [], []
        for stone in self.env.structure_sgf(board):
            color, (x, y) = stone
            colors.append(color)
            actions.append([x * self.args.columns + y for x, y in self.augment_coordinate(x, y)])
        boards = []
        if len(actions) < 1:
            return boards
        for i in range(len(actions[0])):
            boards.append(
                self.env.semicolon.join([colors[j] + self.env.hex_action(actions[j][i]) for j in range(len(colors))]))
        return boards

    def rot90(self, x, y):
        return y, self.args.columns - x - 1

    def fliplr(self, x, y):
        return x, self.args.columns - y - 1

    def augment_coordinate(self, x, y):
        coordinates = []
        for _ in range(4):
            x, y = self.rot90(x, y)
            coordinates.append((x, y))
            coordinates.append(self.fliplr(x, y))
        return coordinates

    def augment_policy(self, policy):
        policies = []
        original = policy.reshape(self.args.rows, self.args.columns)
        for i in range(1, 5):
            p = numpy.rot90(original, -i)
            policies.append(p.flatten())
            policies.append(numpy.fliplr(p).flatten())
        return policies
