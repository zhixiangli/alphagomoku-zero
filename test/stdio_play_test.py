#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

from alphazero.gomoku_stdio import _parse_move


class TestStdioMoveParsing(unittest.TestCase):
    def test_parse_compact_coordinate(self):
        self.assertEqual(_parse_move("E5", rows=9, columns=9), 4 * 9 + 4)

    def test_parse_spaced_coordinate(self):
        self.assertEqual(_parse_move("O 15", rows=15, columns=15), 14 * 15 + 14)

    def test_parse_rejects_out_of_bounds(self):
        self.assertIsNone(_parse_move("P1", rows=15, columns=15))
        self.assertIsNone(_parse_move("A16", rows=15, columns=15))


if __name__ == "__main__":
    unittest.main()
