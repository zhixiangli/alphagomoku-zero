#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import logging

import numpy

from alphazero.game import Game


class ChessType:
    BLACK = "B"
    WHITE = "W"
    EMPTY = "."


class Connect4Game(Game):
    def __init__(self, args):
        self.args = args
        self.semicolon = ";"
        self.directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        self._total = args.rows * args.columns
        self._cols = args.columns
        self._rows = args.rows
        self._n = args.n_in_row

    def next_player(self, player):
        assert player != ChessType.EMPTY
        return ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE

    def _column_heights(self, board):
        heights = [0] * self._cols
        if not board:
            return heights
        for stone in board.split(self.semicolon):
            if not stone:
                continue
            col = int(stone[3], 16)
            heights[col] += 1
        return heights

    def _drop_row(self, heights, col):
        return self._rows - heights[col] - 1

    def next_state(self, board, action, player):
        col = action % self._cols
        heights = self._column_heights(board)
        if heights[col] >= self._rows:
            raise ValueError(f"Column {col} is full")
        row = self._drop_row(heights, col)
        stone = "%s[%x%x]" % (player, row, col)
        next_p = self.next_player(player)
        if board:
            return board + self.semicolon + stone, next_p
        return stone, next_p

    def is_terminal_state(self, board, action, player):
        if not board:
            return None

        rows = self._rows
        cols = self._cols
        n = self._n

        player_positions = set()
        total_stones = 0
        for stone in board.split(self.semicolon):
            if not stone:
                continue
            x = int(stone[2], 16)
            y = int(stone[3], 16)
            if 0 <= x < rows and 0 <= y < cols:
                total_stones += 1
                if stone[0] == player:
                    player_positions.add((x, y))

        ax, ay = divmod(action, cols)
        if (ax, ay) not in player_positions:
            heights = self._column_heights(board)
            ay = action % cols
            ax = self._drop_row(heights, ay) + 1

        has_stone = player_positions.__contains__
        for dx, dy in self.directions:
            count = 1
            x, y = ax + dx, ay + dy
            while 0 <= x < rows and 0 <= y < cols and has_stone((x, y)):
                count += 1
                x += dx
                y += dy
            x, y = ax - dx, ay - dy
            while 0 <= x < rows and 0 <= y < cols and has_stone((x, y)):
                count += 1
                x -= dx
                y -= dy
            if count >= n:
                return player

        if total_stones == self._total:
            return Game.DRAW
        return None

    def get_initial_state(self):
        return "", ChessType.BLACK

    def available_actions(self, board):
        heights = self._column_heights(board)
        actions = []
        for col in range(self._cols):
            if heights[col] < self._rows:
                row = self._drop_row(heights, col)
                actions.append(row * self._cols + col)
        return actions

    def log_status(self, board, counts, actions):
        logging.info("board status: %s", board)

        view = self.to_board(board)
        for row in view:
            logging.info("".join(row))

        visited = numpy.zeros(self._total)
        visited[actions] = counts
        visited = visited.reshape(self._rows, self._cols)
        for row in visited:
            logging.info(",".join(["%3d" % r for r in row]))

    def get_canonical_form(self, board, player):
        rows = self._rows
        cols = self._cols
        feature = numpy.zeros((rows, cols, 2))
        if board:
            opponent = self.next_player(player)
            for stone in board.split(self.semicolon):
                if not stone:
                    continue
                color = stone[0]
                x = int(stone[2], 16)
                y = int(stone[3], 16)
                if color == player:
                    feature[x, y, 0] = 1
                elif color == opponent:
                    feature[x, y, 1] = 1
        return feature

    def to_board(self, sgf):
        board = numpy.full((self._rows, self._cols), ChessType.EMPTY, dtype="U1")
        for stone in self.structure_sgf(sgf):
            color, (x, y) = stone
            board[x, y] = color
        return board

    def structure_sgf(self, sgf):
        return [(s[0], (int(s[2], 16), int(s[3], 16))) for s in sgf.split(";") if s]

    def hex_action(self, action):
        return "[%x%x]" % (action // self._cols, action % self._cols)

    def dec_action(self, stone):
        return int(stone[2], 16), int(stone[3], 16)

    def augment_samples(self, samples):
        augmented = []
        for sample in samples:
            board, policy, value = sample
            boards = self.augment_board(board)
            policies = self.augment_policy(policy)
            augmented.extend([(b, p, value) for b, p in zip(boards, policies)])
        return augmented

    def augment_board(self, board):
        return [board, numpy.flip(board, axis=1)]

    def augment_policy(self, policy):
        original = policy.reshape(self.args.rows, self.args.columns)
        flipped = numpy.flip(original, axis=1)
        return [original.reshape(-1), flipped.reshape(-1)]
