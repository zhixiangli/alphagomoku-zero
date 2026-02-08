#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import argparse
import unittest
from unittest.mock import patch, MagicMock

from alphazero.trainer import add_alphazero_args, extract_alphazero_args, run_training, setup_logging


class TestAddAlphaZeroArgs(unittest.TestCase):

    def test_adds_all_common_args(self):
        """All AlphaZeroConfig fields are present as CLI arguments."""
        parser = argparse.ArgumentParser()
        add_alphazero_args(parser)
        args = parser.parse_args([])
        expected = {
            'save_checkpoint_path', 'sample_pool_file', 'persist_interval',
            'batch_size', 'epochs', 'lr', 'l2',
            'conv_filters', 'conv_kernel', 'residual_block_num',
            'simulation_num', 'history_num', 'c_puct',
            'max_sample_pool_size', 'temp_step',
        }
        self.assertEqual(expected, set(vars(args).keys()))

    def test_cli_overrides(self):
        """CLI values override defaults."""
        parser = argparse.ArgumentParser()
        add_alphazero_args(parser)
        args = parser.parse_args(['-batch_size', '64', '-lr', '0.001'])
        self.assertEqual(args.batch_size, 64)
        self.assertAlmostEqual(args.lr, 0.001)


class TestExtractAlphaZeroArgs(unittest.TestCase):

    def test_returns_correct_keys(self):
        """Extracted dict contains exactly the AlphaZeroConfig fields."""
        parser = argparse.ArgumentParser()
        add_alphazero_args(parser)
        cli_args = parser.parse_args([])
        result = extract_alphazero_args(cli_args)
        expected_keys = {
            'simulation_num', 'c_puct', 'temp_step', 'batch_size',
            'epochs', 'max_sample_pool_size', 'persist_interval',
            'history_num', 'lr', 'l2', 'conv_filters', 'conv_kernel',
            'residual_block_num', 'save_checkpoint_path', 'sample_pool_file',
        }
        self.assertEqual(expected_keys, set(result.keys()))

    def test_preserves_values(self):
        """Extracted values match parsed CLI values."""
        parser = argparse.ArgumentParser()
        add_alphazero_args(parser)
        cli_args = parser.parse_args(['-batch_size', '128'])
        result = extract_alphazero_args(cli_args)
        self.assertEqual(result['batch_size'], 128)


class TestRunTraining(unittest.TestCase):

    def test_calls_module_and_starts_training(self):
        """run_training wires module, loads checkpoint, and starts."""
        mock_module = MagicMock()
        mock_trainer = MagicMock()
        mock_module.create_trainer.return_value = mock_trainer

        config = MagicMock()
        config.save_checkpoint_path = '/tmp/model'

        game_class = type('FakeGame', (), {})

        run_training(mock_module, game_class, config)

        mock_module.create_trainer.assert_called_once_with(game_class, config)
        mock_trainer.nnet.load_checkpoint.assert_called_once_with('/tmp/model')
        mock_trainer.start.assert_called_once()


class TestSetupLogging(unittest.TestCase):

    @patch('alphazero.trainer.logging')
    def test_adds_handlers(self, mock_logging):
        """setup_logging adds file and console handlers."""
        mock_root = MagicMock()
        mock_logging.getLogger.return_value = mock_root
        setup_logging('/tmp/test.log')
        self.assertEqual(mock_root.addHandler.call_count, 2)
        mock_root.setLevel.assert_called_once()


class TestGomokuTrainerMain(unittest.TestCase):

    def test_main_is_importable(self):
        """gomoku.trainer.main exists and is callable."""
        from gomoku.trainer import main
        self.assertTrue(callable(main))

    @patch('gomoku.trainer.run_training')
    @patch('gomoku.trainer.setup_logging')
    def test_main_wires_correctly(self, mock_logging, mock_run):
        """main() parses args, builds config, and calls run_training."""
        from gomoku.trainer import main
        with patch('sys.argv', ['trainer', '-rows', '9', '-columns', '9', '-n_in_row', '3']):
            main()
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        # args[2] is the config
        config = args[2]
        self.assertEqual(config.rows, 9)
        self.assertEqual(config.columns, 9)
        self.assertEqual(config.n_in_row, 3)


class TestTicTacToeTrainerMain(unittest.TestCase):

    def test_main_is_importable(self):
        """tictactoe.trainer.main exists and is callable."""
        from tictactoe.trainer import main
        self.assertTrue(callable(main))

    @patch('tictactoe.trainer.run_training')
    @patch('tictactoe.trainer.setup_logging')
    def test_main_wires_correctly(self, mock_logging, mock_run):
        """main() parses args, builds config, and calls run_training."""
        from tictactoe.trainer import main
        with patch('sys.argv', ['trainer', '-rows', '3', '-columns', '3', '-n_in_row', '3']):
            main()
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        # args[2] is the config
        config = args[2]
        self.assertEqual(config.rows, 3)
        self.assertEqual(config.columns, 3)
        self.assertEqual(config.n_in_row, 3)


if __name__ == '__main__':
    unittest.main()
