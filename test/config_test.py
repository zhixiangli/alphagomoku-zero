#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import unittest

from alphazero.config import AlphaZeroConfig
from gomoku_9_9.config import GomokuConfig


class TestAlphaZeroConfig(unittest.TestCase):
    def test_default_values(self):
        config = AlphaZeroConfig(rows=10, columns=10)
        self.assertEqual(config.rows, 10)
        self.assertEqual(config.columns, 10)
        self.assertEqual(config.simulation_num, 1000)
        self.assertEqual(config.c_puct, 1.0)
        self.assertEqual(config.temp_step, 2)
        self.assertEqual(config.batch_size, 1024)
        self.assertEqual(config.epochs, 20)
        self.assertEqual(config.train_interval, 20)

    def test_action_space_size(self):
        config = AlphaZeroConfig(rows=15, columns=15)
        self.assertEqual(config.action_space_size, 225)

        config = AlphaZeroConfig(rows=3, columns=3)
        self.assertEqual(config.action_space_size, 9)

    def test_custom_values(self):
        config = AlphaZeroConfig(rows=8, columns=8, simulation_num=100, c_puct=2.0)
        self.assertEqual(config.rows, 8)
        self.assertEqual(config.columns, 8)
        self.assertEqual(config.simulation_num, 100)
        self.assertEqual(config.c_puct, 2.0)


class TestGomokuConfig(unittest.TestCase):
    def test_inherits_alphazero_config(self):
        config = GomokuConfig()
        self.assertIsInstance(config, AlphaZeroConfig)

    def test_default_gomoku_values(self):
        config = GomokuConfig()
        self.assertEqual(config.rows, 9)
        self.assertEqual(config.columns, 9)
        self.assertEqual(config.n_in_row, 5)
        self.assertEqual(config.action_space_size, 81)

    def test_game_specific_paths(self):
        config = GomokuConfig()
        self.assertEqual(config.save_checkpoint_path, "./gomoku_9_9/data/model")
        self.assertEqual(config.sample_pool_file, "./gomoku_9_9/data/samples.pkl")

    def test_custom_gomoku_values(self):
        config = GomokuConfig(rows=3, columns=3, n_in_row=2)
        self.assertEqual(config.rows, 3)
        self.assertEqual(config.columns, 3)
        self.assertEqual(config.n_in_row, 2)
        self.assertEqual(config.action_space_size, 9)

    def test_has_common_config_fields(self):
        config = GomokuConfig()
        self.assertEqual(config.simulation_num, 400)
        self.assertEqual(config.c_puct, 1.5)
        self.assertEqual(config.batch_size, 512)


if __name__ == "__main__":
    unittest.main()
