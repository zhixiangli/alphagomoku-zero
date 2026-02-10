#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import logging

import numpy

from alphazero.game import Game


class ChessType:
    BLACK = "B"
    WHITE = "W"
    EMPTY = "."


class GomokuGame(Game):
    def __init__(self, args):
        self.args = args
        self.semicolon = ";"
        self.directions = ((1, 1), (1, -1), (0, 1), (1, 0))
        self._total = args.rows * args.columns
        self._cols = args.columns
        self._rows = args.rows
        self._n = args.n_in_row

    def next_player(self, player):
        assert player != ChessType.EMPTY
        return ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE

    def next_state(self, board, action, player):
        cols = self._cols
        stone = "%s[%x%x]" % (player, action // cols, action % cols)
        next_p = (
            ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE
        )
        if board:
            return board + ";" + stone, next_p
        return stone, next_p

    def is_terminal_state(self, board, action, player):
        if not board:
            return None

        rows = self._rows
        cols = self._cols
        n = self._n

        # Build set of current player's positions and count valid stones
        player_positions = set()
        total_stones = 0
        for stone in board.split(";"):
            x = int(stone[2], 16)
            y = int(stone[3], 16)
            if 0 <= x < rows and 0 <= y < cols:
                total_stones += 1
                if stone[0] == player:
                    player_positions.add((x, y))

        # Check win: count consecutive stones through the action position
        ax, ay = divmod(action, cols)
        pos_in = player_positions.__contains__

        for dx, dy in self.directions:
            count = 1
            # Positive direction
            x, y = ax + dx, ay + dy
            while 0 <= x < rows and 0 <= y < cols and pos_in((x, y)):
                count += 1
                x += dx
                y += dy
            # Negative direction
            x, y = ax - dx, ay - dy
            while 0 <= x < rows and 0 <= y < cols and pos_in((x, y)):
                count += 1
                x -= dx
                y -= dy
            if count >= n:
                return player

        # Check draw
        if total_stones == self._total:
            return Game.DRAW

        return None

    def get_initial_state(self):
        return "", ChessType.BLACK

    def available_actions(self, board):
        total = self._total
        if not board:
            return list(range(total))
        cols = self._cols
        occupied = set()
        for stone in board.split(";"):
            occupied.add(int(stone[2], 16) * cols + int(stone[3], 16))
        return [i for i in range(total) if i not in occupied]

    def log_status(self, board, counts, actions):
        logging.info("board status: %s", board)

        board = self.to_board(board)
        for row in board:
            logging.info("".join(row))

        visited = numpy.zeros(self._total)
        visited[actions] = counts
        visited = visited.reshape(self._rows, self._cols)
        for row in visited:
            logging.info(",".join(["%3d" % r for r in row]))

    def get_canonical_form(self, board, player):
        """Returns board tensor from the perspective of the current player.

        Channel 0: current player's stones
        Channel 1: opponent's stones
        """
        rows = self._rows
        cols = self._cols
        feature = numpy.zeros((rows, cols, 2))
        if board:
            opponent = (
                ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE
            )
            for stone in board.split(";"):
                c = stone[0]
                x = int(stone[2], 16)
                y = int(stone[3], 16)
                if c == player:
                    feature[x, y, 0] = 1
                elif c == opponent:
                    feature[x, y, 1] = 1
        return feature

    def to_board(self, sgf):
        board = numpy.full((self._rows, self._cols), ChessType.EMPTY, dtype="U1")
        for stone in self.structure_sgf(sgf):
            color, (x, y) = stone
            board[x, y] = color
        return board

    def structure_sgf(self, sgf):
        return [
            (s[0], (int(s[2], 16), int(s[3], 16)))
            for s in sgf.split(";")
            if s
        ]

    def hex_action(self, action):
        return "[%x%x]" % (action // self._cols, action % self._cols)

    def dec_action(self, stone):
        return int(stone[2], 16), int(stone[3], 16)

    # Data augmentation methods (merged from GomokuRL)

    def augment_samples(self, samples):
        augmented = []
        for sample in samples:
            board, policy, value = sample
            boards = self.augment_board(board)
            policies = self.augment_policy(policy)
            augmented.extend([(b, p, value) for b, p in zip(boards, policies)])
        return augmented

    def augment_board(self, board):
        """Apply D4 symmetry group transformations to a board tensor."""
        results = []
        for k in range(1, 5):
            rotated = numpy.rot90(board, -k, axes=(0, 1))
            results.append(rotated)
            results.append(numpy.flip(rotated, axis=1))
        return results

    def augment_policy(self, policy):
        original = policy.reshape(self.args.rows, self.args.columns)
        rotations = numpy.stack([numpy.rot90(original, -i) for i in range(1, 5)])
        flipped = numpy.flip(rotations, axis=2)
        result = numpy.empty((8, self.args.rows * self.args.columns))
        result[0::2] = rotations.reshape(4, -1)
        result[1::2] = flipped.reshape(4, -1)
        return list(result)
