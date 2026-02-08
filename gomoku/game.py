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
        board_map = {}
        for stone in stones:
            board_map[self.dec_action(stone)] = stone[0]
        assert len(board_map) == len(stones)
        if any(self.is_win(action, player, direction, board_map) for direction in self.directions):
            return player
        if len(stones) == self.args.rows * self.args.columns:
            return ChessType.EMPTY
        return None

    def get_initial_state(self):
        return '', ChessType.BLACK

    def available_actions(self, board):
        stones = set([stone[1:] for stone in board.split(self.semicolon)])
        return [i for i in range(self.args.rows * self.args.columns) if self.hex_action(i) not in stones]

    def log_status(self, board, counts, actions):
        logging.info("board status: %s", board)

        board = self.to_board(board)
        for row in board:
            logging.info(''.join(row))

        visited = numpy.zeros(self.args.rows * self.args.columns)
        visited[actions] = counts
        for i in range(self.args.rows):
            row = visited[i * self.args.columns:(i + 1) * self.args.columns]
            logging.info(','.join(["%3d" % r for r in row]))

    def compute_reward(self, winner, player):
        if winner == ChessType.EMPTY:
            return 0
        return 1 if player == winner else -1

    def to_board(self, sgf):
        board = [[ChessType.EMPTY] * self.args.columns for _ in range(self.args.rows)]
        for stone in self.structure_sgf(sgf):
            color, (x, y) = stone
            board[x][y] = color
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

    def is_win(self, action, player, direction, board_map):
        dx, dy = direction
        x, y = action // self.args.columns, action % self.args.columns
        cnt = 0
        i, j = x, y
        while 0 <= i < self.args.rows and 0 <= j < self.args.columns and board_map.get((i, j)) == player:
            cnt += 1
            i += dx
            j += dy
        i, j = x - dx, y - dy
        while 0 <= i < self.args.rows and 0 <= j < self.args.columns and board_map.get((i, j)) == player:
            cnt += 1
            i -= dx
            j -= dy
        return cnt >= self.args.n_in_row

    # Data augmentation methods (merged from GomokuRL)

    def augment_samples(self, samples):
        augmented = []
        samples += self.reverse_color(samples)
        for sample in samples:
            board, player, policy, value = sample
            boards = self.augment_board(board)
            policies = self.augment_policy(policy)
            for i in range(len(boards)):
                augmented.append((boards[i], player, policies[i], value))
        return augmented

    def augment_board(self, board):
        colors, actions = [], []
        for stone in self.structure_sgf(board):
            color, (x, y) = stone
            colors.append(color)
            actions.append([x * self.args.columns + y for x, y in self.augment_coordinate(x, y)])
        boards = []
        if len(actions) < 1:
            return boards
        for i in range(len(actions[0])):
            boards.append(
                self.semicolon.join([colors[j] + self.hex_action(actions[j][i]) for j in range(len(colors))]))
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

    def reverse_color(self, samples):
        reversed_samples = []
        for sample in samples:
            board, player, policy, value = sample
            reversed_board = ''.join(list(
                (map(lambda c: self.next_player(c) if c == ChessType.WHITE or c == ChessType.BLACK else c, board))))
            reversed_player = self.next_player(player)
            reversed_samples.append((reversed_board, reversed_player, policy, value))
        return reversed_samples
