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
        self.directions = ((1, 1), (1, -1), (0, 1), (1, 0))
        # Pre-compute hex action strings for all board positions
        cols = args.columns
        total = args.rows * cols
        self._action_hex = [
            "[%x%x]" % (i // cols, i % cols) for i in range(total)
        ]

    def next_player(self, player):
        assert player != ChessType.EMPTY
        return ChessType.BLACK if player == ChessType.WHITE else ChessType.WHITE

    def next_state(self, board, action, player):
        stone = player + self._action_hex[action]
        return (board + ';' + stone if board else stone), self.next_player(player)

    def is_terminal_state(self, board, action, player):
        if not board:
            return None
        # Parse board into a set of the current player's positions
        player_positions = set()
        stone_count = 0
        for stone in board.split(';'):
            stone_count += 1
            if stone[0] == player:
                player_positions.add((int(stone[2], 16), int(stone[3], 16)))
        # Check win around the action in all four directions
        cols = self.args.columns
        ax, ay = action // cols, action % cols
        n = self.args.n_in_row
        rows, columns = self.args.rows, cols
        for dx, dy in self.directions:
            count = 1
            for i in range(1, n):
                nx, ny = ax + i * dx, ay + i * dy
                if 0 <= nx < rows and 0 <= ny < columns and (nx, ny) in player_positions:
                    count += 1
                else:
                    break
            for i in range(1, n):
                nx, ny = ax - i * dx, ay - i * dy
                if 0 <= nx < rows and 0 <= ny < columns and (nx, ny) in player_positions:
                    count += 1
                else:
                    break
            if count >= n:
                return player
        if stone_count == rows * columns:
            return Game.DRAW
        return None

    def get_initial_state(self):
        return '', ChessType.BLACK

    def available_actions(self, board):
        total = self.args.rows * self.args.columns
        if not board:
            return list(range(total))
        cols = self.args.columns
        occupied = set()
        for stone in board.split(';'):
            occupied.add(int(stone[2], 16) * cols + int(stone[3], 16))
        return [i for i in range(total) if i not in occupied]

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

    def get_canonical_form(self, board, player):
        """Returns board tensor from the perspective of the current player.

        Channel 0: current player's stones
        Channel 1: opponent's stones
        """
        feature = numpy.zeros((self.args.rows, self.args.columns, 2), dtype=numpy.float32)
        if board:
            for stone in board.split(';'):
                x, y = int(stone[2], 16), int(stone[3], 16)
                feature[x, y, 0 if stone[0] == player else 1] = 1
        return feature

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
        return self._action_hex[action]

    def dec_action(self, stone):
        return int(stone[2], 16), int(stone[3], 16)

    def is_win(self, action, player, direction, board_array):
        dx, dy = direction
        cols = self.args.columns
        x, y = action // cols, action % cols
        n = self.args.n_in_row
        rows, columns = self.args.rows, cols
        count = 1
        for i in range(1, n):
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < rows and 0 <= ny < columns and board_array[nx, ny] == player:
                count += 1
            else:
                break
        for i in range(1, n):
            nx, ny = x - i * dx, y - i * dy
            if 0 <= nx < rows and 0 <= ny < columns and board_array[nx, ny] == player:
                count += 1
            else:
                break
        return count >= n

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
        rotations = numpy.stack([numpy.rot90(original, -i) for i in range(1, 5)])
        flipped = numpy.flip(rotations, axis=2)
        result = numpy.empty((8, self.args.rows * self.args.columns))
        result[0::2] = rotations.reshape(4, -1)
        result[1::2] = flipped.reshape(4, -1)
        return list(result)

