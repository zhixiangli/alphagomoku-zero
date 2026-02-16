#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

from connect4.stdio_play import (
    _parse_move as _parse_connect4_move,
    _resolve_action as _resolve_connect4_action,
)


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
