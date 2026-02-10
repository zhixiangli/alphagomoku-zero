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
    """Connect Four game (4-in-a-row on a 6x7 board with gravity).

    Players take turns dropping pieces into columns.  Pieces fall to the
    lowest available row.  The first player to get ``n_in_row`` consecutive
    pieces (horizontally, vertically, or diagonally) wins.

    Actions are column indices (0 to ``columns - 1``).
    """

    def __init__(self, args):
        self.args = args
        self.directions = ((1, 1), (1, -1), (0, 1), (1, 0))
        self._total = args.rows * args.columns
        self._cols = args.columns
        self._rows = args.rows
        self._n = args.n_in_row

    def next_player(self, player):
        assert player != ChessType.EMPTY
        return ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE

    def _find_landing_row(self, board, col):
        """Find the lowest empty row in *col* (gravity)."""
        occupied_rows = set()
        if board:
            for stone in board.split(";"):
                y = int(stone[3], 16)
                if y == col:
                    occupied_rows.add(int(stone[2], 16))
        for row in range(self._rows - 1, -1, -1):
            if row not in occupied_rows:
                return row
        return None

    def next_state(self, board, action, player):
        col = action
        row = self._find_landing_row(board, col)
        stone = "%s[%x%x]" % (player, row, col)
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

        # Build set of current player's positions and count total stones
        player_positions = set()
        total_stones = 0
        for stone in board.split(";"):
            x = int(stone[2], 16)
            y = int(stone[3], 16)
            if 0 <= x < rows and 0 <= y < cols:
                total_stones += 1
                if stone[0] == player:
                    player_positions.add((x, y))

        # The last stone in the board string is the most recently placed one
        stones = board.split(";")
        last_stone = stones[-1]
        ax = int(last_stone[2], 16)
        ay = int(last_stone[3], 16)

        has_stone = player_positions.__contains__

        for dx, dy in self.directions:
            count = 1
            # Positive direction
            x, y = ax + dx, ay + dy
            while 0 <= x < rows and 0 <= y < cols and has_stone((x, y)):
                count += 1
                x += dx
                y += dy
            # Negative direction
            x, y = ax - dx, ay - dy
            while 0 <= x < rows and 0 <= y < cols and has_stone((x, y)):
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
        if not board:
            return list(range(self._cols))
        col_count = [0] * self._cols
        for stone in board.split(";"):
            y = int(stone[3], 16)
            col_count[y] += 1
        return [col for col in range(self._cols) if col_count[col] < self._rows]

    def log_status(self, board, counts, actions):
        logging.info("board status: %s", board)

        grid = self.to_board(board)
        for row in grid:
            logging.info("".join(row))

        visited = numpy.zeros(self._cols)
        visited[actions] = counts
        logging.info(",".join(["%3d" % r for r in visited]))

    def get_canonical_form(self, board, player):
        """Returns board tensor from the perspective of the current player.

        Channel 0: current player's pieces
        Channel 1: opponent's pieces
        """
        rows = self._rows
        cols = self._cols
        feature = numpy.zeros((rows, cols, 2))
        if board:
            opponent = (
                ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE
            )
            for stone in board.split(";"):
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
        if sgf:
            for stone in sgf.split(";"):
                if stone:
                    color = stone[0]
                    x = int(stone[2], 16)
                    y = int(stone[3], 16)
                    board[x, y] = color
        return board

    # Data augmentation methods

    def augment_samples(self, samples):
        """Augment training data using horizontal flip symmetry.

        Connect4 has gravity so only left-right mirror is valid
        (unlike Gomoku which uses full D4 symmetry).
        """
        augmented = []
        for sample in samples:
            board, policy, value = sample
            augmented.append(sample)
            # Horizontal flip
            flipped_board = numpy.flip(board, axis=1).copy()
            flipped_policy = policy.copy()
            flipped_policy[: self._cols] = policy[: self._cols][::-1]
            augmented.append((flipped_board, flipped_policy, value))
        return augmented
