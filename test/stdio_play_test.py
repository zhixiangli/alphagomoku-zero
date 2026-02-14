#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

from gomoku_15_15.stdio_play import _parse_move
from connect4.stdio_play import (
    _parse_move as _parse_connect4_move,
    _resolve_action as _resolve_connect4_action,
)


class TestStdioMoveParsing(unittest.TestCase):
    def test_parse_compact_coordinate(self):
        self.assertEqual(_parse_move("E5", rows=9, columns=9), 4 * 9 + 4)

    def test_parse_spaced_coordinate(self):
        self.assertEqual(_parse_move("O 15", rows=15, columns=15), 14 * 15 + 14)

    def test_parse_rejects_out_of_bounds(self):
        self.assertIsNone(_parse_move("P1", rows=15, columns=15))
        self.assertIsNone(_parse_move("A16", rows=15, columns=15))


class TestConnect4StdioMoveParsing(unittest.TestCase):
    def test_parse_column_number(self):
        self.assertEqual(_parse_connect4_move("4", rows=6, columns=7), 3)

    def test_parse_rejects_invalid_column(self):
        self.assertIsNone(_parse_connect4_move("0", rows=6, columns=7))
        self.assertIsNone(_parse_connect4_move("8", rows=6, columns=7))
        self.assertIsNone(_parse_connect4_move("A", rows=6, columns=7))

    def test_resolve_column_to_legal_drop_action(self):
        self.assertEqual(_resolve_connect4_action(3, [35, 36, 37, 38, 39, 40, 41]), 38)

    def test_resolve_rejects_full_or_invalid_column(self):
        self.assertIsNone(_resolve_connect4_action(3, [35, 36, 37, 39, 40, 41]))
        self.assertIsNone(_resolve_connect4_action(None, [35, 36, 37]))


if __name__ == "__main__":
    unittest.main()
