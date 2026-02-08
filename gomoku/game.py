#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import logging

import numpy

from alphazero.game import Game


class ChessType:
    BLACK = 'B'
    WHITE = 'W'
    EMPTY = '.'


class GomokuGame(Game):

    def __init__(self, args):
        self.args = args
        self.semicolon = ';'
        self.directions = [[1, 1], [1, -1], [0, 1], [1, 0]]

    @property
    def action_space_size(self):
        return self.args.rows * self.args.columns

    def next_player(self, player):
        assert player != ChessType.EMPTY
        return ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE

    def next_state(self, board, action, player):
        stone = player + self.hex_action(action)
        return board + self.semicolon + stone if board else stone, self.next_player(player)

    def is_terminal_state(self, board, action, player):
        stones = board.split(self.semicolon)
        if not stones:
            return None
        board_array = numpy.full((self.args.rows, self.args.columns), ChessType.EMPTY, dtype='U1')
        for stone in stones:
            (x, y) = self.dec_action(stone)
            if 0 <= x < self.args.rows and 0 <= y < self.args.columns:
                board_array[x, y] = stone[0]
        if any(self.is_win(action, player, direction, board_array) for direction in self.directions):
            return player
        if int(numpy.sum(board_array != ChessType.EMPTY)) == self.args.rows * self.args.columns:
            return ChessType.EMPTY
        return None

    def get_initial_state(self):
        return '', ChessType.BLACK

    def available_actions(self, board):
        total = self.args.rows * self.args.columns
        occupied = numpy.zeros(total, dtype=bool)
        if board:
            for stone in board.split(self.semicolon):
                x, y = self.dec_action(stone)
                occupied[x * self.args.columns + y] = True
        return numpy.flatnonzero(~occupied).tolist()

    def log_status(self, board, counts, actions):
        logging.info("board status: %s", board)

        board = self.to_board(board)
        for row in board:
            logging.info(''.join(row))

        visited = numpy.zeros(self.args.rows * self.args.columns)
        visited[actions] = counts
        visited = visited.reshape(self.args.rows, self.args.columns)
        for row in visited:
            logging.info(','.join(["%3d" % r for r in row]))

    def compute_reward(self, winner, player):
        if winner == ChessType.EMPTY:
            return 0
        return 1 if player == winner else -1

    def to_board(self, sgf):
        board = numpy.full((self.args.rows, self.args.columns), ChessType.EMPTY, dtype='U1')
        for stone in self.structure_sgf(sgf):
            color, (x, y) = stone
            board[x, y] = color
        return board

    def structure_sgf(self, sgf):
        return [(s[0], self.dec_action(s)) for s in sgf.split(self.semicolon) if s]

    def to_sgf(self, board):
        return self.semicolon.join(
            ["%c%s" % (board[i][j], self.hex_action(i * self.args.columns + j)) for i in range(self.args.rows) for j in
             range(self.args.columns) if board[i][j] != ChessType.EMPTY])

    def hex_action(self, action):
        def dec_to_hex(dec):
            return format(dec, 'x')

        return "[%s%s]" % (dec_to_hex(action // self.args.columns), dec_to_hex(action % self.args.columns))

    def dec_action(self, stone):
        def hex_to_dec(hex):
            return int(hex, 16)

        return hex_to_dec(stone[2]), hex_to_dec(stone[3])

    def is_win(self, action, player, direction, board_array):
        dx, dy = direction
        x, y = action // self.args.columns, action % self.args.columns
        n = self.args.n_in_row
        offsets = numpy.arange(-(n - 1), n)
        xs = x + offsets * dx
        ys = y + offsets * dy
        valid = (xs >= 0) & (xs < self.args.rows) & (ys >= 0) & (ys < self.args.columns)
        xs_safe = numpy.where(valid, xs, 0)
        ys_safe = numpy.where(valid, ys, 0)
        matches = valid & (board_array[xs_safe, ys_safe] == player)
        if numpy.sum(matches) < n:
            return False
        kernel = numpy.ones(n, dtype=int)
        conv = numpy.convolve(matches.astype(int), kernel, mode='valid')
        return bool(numpy.any(conv >= n))

    # Data augmentation methods (merged from GomokuRL)

    def augment_samples(self, samples):
        augmented = []
        samples += self.reverse_color(samples)
        for sample in samples:
            board, player, policy, value = sample
            boards = self.augment_board(board)
            policies = self.augment_policy(policy)
            augmented.extend([(b, player, p, value) for b, p in zip(boards, policies)])
        return augmented

    def augment_board(self, board):
        structured = self.structure_sgf(board)
        if not structured:
            return []
        colors = [s[0] for s in structured]
        coords = numpy.array([(s[1][0], s[1][1]) for s in structured])
        x, y = coords[:, 0], coords[:, 1]
        c = self.args.columns - 1
        aug_x = numpy.stack([y, y, c - x, c - x, c - y, c - y, x, x])
        aug_y = numpy.stack([c - x, x, c - y, y, x, c - x, y, c - y])
        aug_actions = aug_x * self.args.columns + aug_y
        boards = []
        for i in range(aug_actions.shape[0]):
            boards.append(
                self.semicolon.join([colors[j] + self.hex_action(int(aug_actions[i, j])) for j in range(len(colors))]))
        return boards

    def rot90(self, x, y):
        return y, self.args.columns - x - 1

    def fliplr(self, x, y):
        return x, self.args.columns - y - 1

    def augment_coordinate(self, x, y):
        c = self.args.columns - 1
        coords = numpy.array([
            [y, c - x], [y, x],
            [c - x, c - y], [c - x, y],
            [c - y, x], [c - y, c - x],
            [x, y], [x, c - y],
        ])
        return list(map(tuple, coords))

    def augment_policy(self, policy):
        original = policy.reshape(self.args.rows, self.args.columns)
        rotations = numpy.array([numpy.rot90(original, -i) for i in range(1, 5)])
        flipped = numpy.flip(rotations, axis=2)
        result = numpy.empty((8, self.args.rows * self.args.columns))
        result[0::2] = rotations.reshape(4, -1)
        result[1::2] = flipped.reshape(4, -1)
        return list(result)

    def reverse_color(self, samples):
        reversed_samples = []
        for sample in samples:
            board, player, policy, value = sample
            reversed_board = ''.join(list(
                (map(lambda c: self.next_player(c) if c == ChessType.WHITE or c == ChessType.BLACK else c, board))))
            reversed_player = self.next_player(player)
            reversed_samples.append((reversed_board, reversed_player, policy, value))
        return reversed_samples
